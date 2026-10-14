"""Shared validated inference and decision pipeline used by API and streaming."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from edgesight.decision import QualityRouter
from edgesight.inference.engine import InferenceEngine
from edgesight.inference.preprocess import ImagePreprocessor
from edgesight.schemas import InferenceMode


class InspectionPipeline:
    """Convert model outputs into the stable manufacturing decision contract."""

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        detector_threshold: float = 0.7,
        anomaly_threshold: float = 0.5,
        max_pixels: int = 25_000_000,
        category: str = "demo",
        camera_id: str = "api",
        human_review_enabled: bool = True,
        preprocessor: ImagePreprocessor | None = None,
        model_version: str | None = None,
        model_hashes: dict[str, str] | None = None,
        runtime: str | None = None,
        artifact_directory: str | Path | None = None,
        anomaly_threshold_source: str = "configured",
    ) -> None:
        self.engine = engine
        self.detector_threshold = detector_threshold
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_threshold_source = anomaly_threshold_source
        self.max_pixels = max_pixels
        self.category = category
        self.camera_id = camera_id
        self.human_review_enabled = human_review_enabled
        self.preprocessor = preprocessor or ImagePreprocessor(max_pixels=max_pixels)
        self._model_version = model_version
        self._model_hashes = dict(model_hashes or {})
        self.runtime = runtime or ("synthetic" if engine.mode is InferenceMode.DEMO else "pytorch")
        self.artifact_directory = (
            Path(artifact_directory) if artifact_directory is not None else None
        )
        self.router = QualityRouter(
            detection_acceptance=detector_threshold,
            anomaly_threshold=anomaly_threshold,
            human_review_enabled=human_review_enabled,
        )

    def is_ready(self) -> bool:
        return all(self.readiness_checks().values())

    def readiness_checks(self) -> dict[str, bool]:
        models_ready = self.engine.mode is InferenceMode.DEMO or (
            self.engine.detector is not None or self.engine.anomaly_detector is not None
        )
        return {"model": models_ready, "preprocess": self.preprocessor is not None}

    def model_info(self) -> dict[str, Any]:
        return {
            "mode": self.engine.mode.value,
            "model_version": self._model_version
            or (
                "demo-deterministic-v1"
                if self.engine.mode is InferenceMode.DEMO
                else "configured-real-model"
            ),
            "model_hashes": self._model_hashes,
            "runtime": self.runtime,
            "preprocessing_version": self.preprocessor.VERSION,
            "detector_threshold": self.detector_threshold,
            "anomaly_threshold": self.anomaly_threshold,
            "anomaly_threshold_source": self.anomaly_threshold_source,
        }

    def predict(
        self,
        payload: bytes,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        prepared = self.preprocessor.process(
            payload, require_tensor=self.engine.mode is not InferenceMode.DEMO
        )
        model_input = payload if self.engine.mode is InferenceMode.DEMO else prepared
        inference = self.engine.predict(model_input, source=filename)
        detections = list(inference.detections)
        best = max(detections, key=lambda item: item.confidence, default=None)
        anomaly_score = inference.anomaly.score if inference.anomaly else None
        routed = self.router.decide(inference)
        detector_trigger = best is not None and best.confidence >= self.detector_threshold

        sample_id = hashlib.sha256(payload).hexdigest()[:24]
        artifact_references = self._save_artifacts(
            sample_id,
            prepared.pil,
            detections,
            inference.anomaly.map if inference.anomaly else None,
        )
        return {
            "sample_id": sample_id,
            "camera_id": metadata.get("camera_id", self.camera_id),
            "frame_id": metadata.get("frame_id"),
            "timestamp": metadata.get("captured_at", datetime.now(UTC).isoformat()),
            "decision": routed.decision.value,
            "decision_reason": routed.reason,
            "decision_confidence": routed.confidence,
            "category": metadata.get("category", self.category),
            "defect_class": best.label if detector_trigger and best else None,
            "detector_confidence": best.confidence if best else None,
            "anomaly_score": anomaly_score,
            "anomaly_threshold": self.anomaly_threshold,
            "anomaly_threshold_source": self.anomaly_threshold_source,
            "detector_threshold": self.detector_threshold,
            "bounding_boxes": [asdict(item) for item in detections],
            "heatmap_available": bool(inference.anomaly and inference.anomaly.map is not None),
            "artifact_references": artifact_references,
            "model_version": self.model_info()["model_version"],
            "model_hashes": self._model_hashes,
            "preprocessing_version": self.preprocessor.VERSION,
            "inference_latency_ms": inference.latency_ms,
            "error": (
                {
                    "code": "model_execution_failed",
                    "models": dict(routed.metadata["failures"]),
                }
                if "failures" in routed.metadata
                else None
            ),
            "review_required": routed.review_required,
            "mode": inference.mode.value,
            "synthetic": inference.mode.value == "demo",
        }

    def _save_artifacts(
        self,
        sample_id: str,
        image: Image.Image,
        detections: list[Any],
        anomaly_map: Any | None,
    ) -> dict[str, str]:
        if self.artifact_directory is None:
            return {}
        destination = self.artifact_directory / sample_id
        destination.mkdir(parents=True, exist_ok=True)
        original_path = destination / "original.png"
        annotated_path = destination / "annotated.jpg"
        image.save(original_path)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for detection in detections:
            draw.rectangle(detection.xyxy, outline=(255, 32, 32), width=3)
            draw.text(
                (detection.xyxy[0], max(0, detection.xyxy[1] - 12)),
                f"{detection.label or detection.class_id} {detection.confidence:.2f}",
                fill=(255, 32, 32),
            )
        annotated.save(annotated_path, quality=90)
        references = {
            "original": str(original_path),
            "annotated": str(annotated_path),
        }
        if anomaly_map is not None:
            value = anomaly_map
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = np.asarray(value, dtype=np.float32).squeeze()
            if array.ndim == 2:
                low, high = float(array.min()), float(array.max())
                normalized = (array - low) / (high - low) if high > low else np.zeros_like(array)
                heat = Image.fromarray((normalized * 255).astype(np.uint8)).resize(
                    image.size, Image.Resampling.BILINEAR
                )
                red = Image.new("RGB", image.size, (255, 0, 0))
                tinted = Image.composite(red, image, heat)
                overlay = Image.blend(image, tinted, 0.45)
                heatmap_path = destination / "heatmap.jpg"
                overlay.save(heatmap_path, quality=90)
                references["heatmap"] = str(heatmap_path)
        return references
