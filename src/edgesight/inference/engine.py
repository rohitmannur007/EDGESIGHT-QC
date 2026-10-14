"""Stable inference contract for deterministic demos and real models."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Protocol

from edgesight.schemas import AnomalyResult, Detection, InferenceMode, InferenceResult

from .preprocess import ImagePreprocessor, PreprocessedImage


class Detector(Protocol):
    def predict(self, image: Any) -> tuple[Detection, ...]: ...


class AnomalyDetector(Protocol):
    def predict(self, images: Any) -> tuple[Any, Any]: ...


class RealInferenceUnavailable(RuntimeError):
    pass


def _content_bytes(image: Any) -> bytes:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Inference source not found: {path}")
        return path.read_bytes()
    if isinstance(image, bytes):
        return image
    if hasattr(image, "tobytes"):
        return image.tobytes()
    return repr(image).encode()


class InferenceEngine:
    def __init__(
        self,
        *,
        mode: InferenceMode | str = InferenceMode.REAL,
        detector: Detector | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        anomaly_threshold: float = 0.5,
        preprocessor: ImagePreprocessor | None = None,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.mode = InferenceMode(mode)
        self.detector = detector
        self.anomaly_detector = anomaly_detector
        self.anomaly_threshold = anomaly_threshold
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.model_metadata = dict(model_metadata or {})
        if self.mode is InferenceMode.REAL and detector is None and anomaly_detector is None:
            raise RealInferenceUnavailable(
                "Real inference requires a detector or anomaly model; "
                "select mode='demo' explicitly for synthetic output"
            )

    def predict(self, image: Any, *, source: str | Path | None = None) -> InferenceResult:
        started = time.perf_counter()
        if self.mode is InferenceMode.DEMO:
            detections, anomaly = self._demo_prediction(image)
            failures: dict[str, str] = {}
        else:
            prepared = image if isinstance(image, PreprocessedImage) else self.preprocessor(image)
            failures = {}
            detections = ()
            anomaly = None
            if self.detector is not None:
                try:
                    detections = self.detector.predict(prepared.detector)
                except Exception as exc:
                    failures["detector"] = f"{type(exc).__name__}: {exc}"
            if self.anomaly_detector is not None:
                try:
                    anomaly = self._anomaly_prediction(prepared.tensor)
                except Exception as exc:
                    failures["anomaly"] = f"{type(exc).__name__}: {exc}"
        return InferenceResult(
            source=source,
            mode=self.mode,
            detections=detections,
            anomaly=anomaly,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={
                "synthetic": self.mode is InferenceMode.DEMO,
                "configured_models": {
                    "detector": self.detector is not None,
                    "anomaly": self.anomaly_detector is not None,
                },
                "model_failures": failures,
                **self.model_metadata,
            },
        )

    def _demo_prediction(self, image: Any) -> tuple[tuple[Detection, ...], AnomalyResult]:
        digest = hashlib.sha256(_content_bytes(image)).digest()
        confidence = 0.35 + digest[0] / 255 * 0.6
        score = digest[1] / 255
        width = 32 + digest[2] % 64
        height = 32 + digest[3] % 64
        detection = Detection(
            class_id=digest[4] % 3,
            confidence=confidence,
            xyxy=(10.0, 10.0, float(10 + width), float(10 + height)),
            label=("scratch", "dent", "contamination")[digest[4] % 3],
        )
        return (detection,), AnomalyResult(
            score=score,
            threshold=self.anomaly_threshold,
            metadata={"synthetic": True},
        )

    def _anomaly_prediction(self, tensor: Any) -> AnomalyResult:
        assert self.anomaly_detector is not None
        scores, maps = self.anomaly_detector.predict(tensor)
        score = float(scores[0].item() if hasattr(scores[0], "item") else scores[0])
        anomaly_map = maps[0] if len(maps) else None
        return AnomalyResult(score=score, map=anomaly_map, threshold=self.anomaly_threshold)
