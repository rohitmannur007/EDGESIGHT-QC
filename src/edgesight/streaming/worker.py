"""Validated, deduplicating streaming inference worker."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

from edgesight.api.adapters import run_pipeline
from edgesight.monitoring.loop import image_observations
from edgesight.persistence import EdgeSightStore

from .broker import MessageProducer
from .messages import validate_message


class StreamingWorker:
    def __init__(
        self,
        pipeline: Any,
        store: EdgeSightStore,
        producer: MessageProducer,
        *,
        output_topic: str = "edgesight.results",
        dlq_topic: str = "edgesight.dlq",
        observations_topic: str = "edgesight.observations",
        events_topic: str = "edgesight.monitoring.events",
        message_schema: type[Any] | None = None,
        monitoring_loop: Any | None = None,
        max_attempts: int = 3,
        transient_exceptions: tuple[type[Exception], ...] = (
            TimeoutError,
            ConnectionError,
        ),
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.producer = producer
        self.output_topic = output_topic
        self.dlq_topic = dlq_topic
        self.observations_topic = observations_topic
        self.events_topic = events_topic
        self.message_schema = message_schema
        self.monitoring_loop = monitoring_loop
        self.max_attempts = max_attempts
        self.transient_exceptions = transient_exceptions

    async def handle(self, raw: Mapping[str, Any]) -> str:
        """Process one message and return processed, duplicate, or dead_letter."""
        try:
            message = validate_message(raw, self.message_schema)
        except Exception as exc:
            self._dead_letter(raw, exc)
            return "dead_letter"
        claim = self.store.claim_message(
            message.message_id,
            metadata=message.metadata,
            max_attempts=self.max_attempts,
        )
        if not claim["claimed"]:
            return "duplicate"
        try:
            result_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"edgesight:{message.message_id}"))
            persisted = self.store.get_result(result_id)
            if persisted and persisted["status"] == "complete":
                prediction = persisted["prediction"]
            else:
                target_pipeline = self.pipeline
                if isinstance(self.pipeline, Mapping):
                    category = str((message.metadata or {}).get("category", ""))
                    target_pipeline = self.pipeline.get(category)
                    if target_pipeline is None:
                        raise ValueError(
                            f"no inference pipeline configured for category {category!r}"
                        )
                prediction = await run_pipeline(
                    target_pipeline,
                    message.payload,
                    filename=message.filename,
                    metadata=message.metadata,
                )
                if isinstance(prediction, dict) and message.metadata:
                    prediction = {**prediction, "source_metadata": message.metadata}
                self.store.save_result(
                    status="complete",
                    prediction=prediction,
                    request_id=message.message_id,
                    source_name=message.filename,
                    result_id=result_id,
                )
            self.producer.send(
                self.output_topic,
                {
                    "message_id": message.message_id,
                    "result_id": result_id,
                    "prediction": prediction,
                    "metadata": message.metadata or {},
                },
                key=message.message_id,
            )
            self._observe(message, prediction)
            self.store.finish_message(message.message_id, "complete", result_id=result_id)
            return "processed"
        except Exception as exc:
            transient = isinstance(exc, self.transient_exceptions)
            if transient and int(claim["attempts"]) < self.max_attempts:
                self.store.finish_message(message.message_id, "retry", error=str(exc))
                return "retry"
            result_id = self.store.save_result(
                status="failed",
                error=str(exc),
                request_id=message.message_id,
                source_name=message.filename,
            )
            self.store.finish_message(
                message.message_id, "failed", result_id=result_id, error=str(exc)
            )
            self._dead_letter(raw, exc, result_id=result_id, message_id=message.message_id)
            return "dead_letter"

    def handle_sync(self, raw: Mapping[str, Any]) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.handle(raw))
        raise RuntimeError(
            "handle_sync cannot run inside an active event loop; await handle instead"
        )

    def _dead_letter(
        self,
        raw: Mapping[str, Any],
        error: Exception,
        *,
        result_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self.producer.send(
            self.dlq_topic,
            {
                "message_id": message_id or raw.get("message_id"),
                "result_id": result_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "original": self._safe_metadata(raw),
            },
            key=message_id,
        )

    def _observe(self, message: Any, prediction: Any) -> None:
        try:
            observations: dict[str, Any] = image_observations(message.payload)
        except Exception:
            observations = {}
        if isinstance(prediction, Mapping):
            observations["anomaly"] = prediction.get("anomaly_score")
            observations["class"] = prediction.get("defect_class") or prediction.get("decision")
        observations = {key: value for key, value in observations.items() if value is not None}
        event = {
            "message_id": message.message_id,
            "observations": observations,
            "metadata": message.metadata or {},
        }
        self.producer.send(self.observations_topic, event, key=message.message_id)
        if self.monitoring_loop is not None:
            outcome = self.monitoring_loop.observe(observations, metadata=message.metadata or {})
            if outcome is not None:
                payload = {
                    "message_id": message.message_id,
                    **outcome.as_dict(),
                    "metadata": message.metadata or {},
                }
                self.producer.send(self.events_topic, payload, key=message.message_id)

    @classmethod
    def _safe_metadata(cls, value: Any) -> Any:
        """Remove payload bytes recursively from DLQ records."""
        if isinstance(value, bytes | bytearray | memoryview):
            return {"omitted_bytes": len(value)}
        if isinstance(value, Mapping):
            return {
                str(key): (
                    {"omitted": True}
                    if str(key).lower() in {"payload", "image", "bytes", "data"}
                    else cls._safe_metadata(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [cls._safe_metadata(item) for item in value]
        return value
