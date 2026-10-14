"""Strict loading and provenance tracking for production model artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelRegistryError(RuntimeError):
    """Base error for an invalid production model configuration."""


class ModelConfigurationError(ModelRegistryError):
    """Raised when required model configuration is absent or ambiguous."""


class ModelArtifactError(ModelRegistryError):
    """Raised when a configured model artifact cannot be loaded."""


class UnsupportedRuntimeError(ModelRegistryError):
    """Raised when a requested runtime is not an API execution backend."""


@dataclass(frozen=True)
class ModelArtifact:
    kind: str
    path: Path
    sha256: str
    version: str
    category: str | None = None


@dataclass(frozen=True)
class LoadedModels:
    detector: Any | None
    anomaly_detector: Any | None
    artifacts: tuple[ModelArtifact, ...]
    category: str
    runtime: str
    anomaly_threshold: float | None = None
    anomaly_threshold_source: str = "unconfigured"

    @property
    def model_version(self) -> str:
        if not self.artifacts:
            return "unconfigured"
        return "+".join(f"{item.kind}:{item.version}" for item in self.artifacts)

    @property
    def hashes(self) -> dict[str, str]:
        return {
            item.kind if item.category is None else f"{item.kind}:{item.category}": item.sha256
            for item in self.artifacts
        }


class ModelRegistry:
    """Load only explicitly configured local artifacts, with no fallback."""

    def __init__(
        self,
        *,
        detector_path: str | Path | None,
        anomaly_paths: Mapping[str, str | Path] | None,
        category: str,
        detector_confidence: float = 0.25,
        detector_iou: float = 0.45,
        device: str = "cpu",
        runtime: str = "pytorch",
        base_directory: str | Path | None = None,
    ) -> None:
        self.base_directory = Path(base_directory or ".").resolve()
        self.detector_path = self._resolve(detector_path)
        self.anomaly_paths = {
            name: self._resolve(path) for name, path in (anomaly_paths or {}).items()
        }
        self.category = category
        self.detector_confidence = detector_confidence
        self.detector_iou = detector_iou
        self.device = device
        self.runtime = runtime

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], *, base_directory: str | Path | None = None
    ) -> ModelRegistry:
        models = settings.get("models") or {}
        if not isinstance(models, Mapping):
            raise ModelConfigurationError("'models' must be a mapping")
        thresholds = settings.get("thresholds") or {}
        detector_threshold = (
            thresholds.get("detector", 0.25) if isinstance(thresholds, Mapping) else 0.25
        )
        return cls(
            detector_path=models.get("detector_path"),
            anomaly_paths=models.get("anomaly_models"),
            category=str(settings.get("category", "default")),
            detector_confidence=float(detector_threshold),
            detector_iou=float(settings.get("detector_iou", 0.45)),
            device=str(settings.get("device", "cpu")),
            runtime=str(settings.get("runtime", "pytorch")),
            base_directory=base_directory,
        )

    def load(self) -> LoadedModels:
        if self.runtime != "pytorch":
            raise UnsupportedRuntimeError(
                f"runtime {self.runtime!r} is not executed by the API. "
                "Supported API runtime: 'pytorch' for Ultralytics YOLO and PatchCore. "
                "ONNX and Core ML measurements belong to scripts/benchmark_models.py "
                "and are not implied by this setting."
            )
        detector = None
        anomaly_detector = None
        artifacts: list[ModelArtifact] = []
        if self.detector_path is not None:
            path = self._require_artifact("detector", self.detector_path)
            if path.suffix.lower() not in {".pt", ".pth"}:
                raise UnsupportedRuntimeError(
                    f"detector artifact {path.name} is not a PyTorch checkpoint. "
                    "The API executes runtime 'pytorch' only. Benchmark ONNX or Core ML "
                    "with scripts/benchmark_models.py; that does not change API execution."
                )
            try:
                from edgesight.detector import YoloDetector

                detector = YoloDetector(
                    path,
                    confidence=self.detector_confidence,
                    iou=self.detector_iou,
                    device=self.device,
                )
                # Force artifact parsing during startup so readiness never
                # reports true for corrupt or incompatible weights.
                detector._load()
            except Exception as exc:
                raise ModelArtifactError(f"failed to load detector artifact {path}: {exc}") from exc
            artifacts.append(self._describe("detector", path))

        anomaly_threshold = None
        anomaly_path = self.anomaly_paths.get(self.category)
        if anomaly_path is not None:
            path = self._require_artifact(f"anomaly model for '{self.category}'", anomaly_path)
            try:
                from edgesight.anomaly import PatchCore

                anomaly_detector = PatchCore.load(path, device=self.device)
            except Exception as exc:
                raise ModelArtifactError(
                    f"failed to load anomaly artifact for '{self.category}' at {path}: {exc}"
                ) from exc
            artifacts.append(self._describe("patchcore", path, self.category))
            anomaly_threshold = self._calibrated_threshold(anomaly_detector)
        elif self.anomaly_paths:
            configured = ", ".join(sorted(self.anomaly_paths))
            raise ModelConfigurationError(
                f"no anomaly artifact configured for category '{self.category}'; "
                f"configured categories: {configured}"
            )

        if detector is None and anomaly_detector is None:
            raise ModelConfigurationError(
                "real inference requires detector_path and/or a category-specific anomaly artifact"
            )
        return LoadedModels(
            detector=detector,
            anomaly_detector=anomaly_detector,
            artifacts=tuple(artifacts),
            category=self.category,
            runtime=self.runtime,
            anomaly_threshold=anomaly_threshold,
            anomaly_threshold_source=(
                "unconfigured"
                if anomaly_detector is None
                else "model_calibration"
                if anomaly_threshold is not None
                else "missing_calibration"
            ),
        )

    @staticmethod
    def _calibrated_threshold(model: Any) -> float | None:
        calibration = getattr(model, "calibration", None)
        threshold = calibration.get("threshold") if isinstance(calibration, Mapping) else None
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            return None
        return float(threshold)

    def _resolve(self, path: str | Path | None) -> Path | None:
        if path in (None, ""):
            return None
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self.base_directory / candidate

    @staticmethod
    def _require_artifact(kind: str, path: Path) -> Path:
        if not path.is_file():
            raise ModelArtifactError(f"configured {kind} artifact not found: {path}")
        return path

    @staticmethod
    def _describe(kind: str, path: Path, category: str | None = None) -> ModelArtifact:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ModelArtifact(
            kind=kind,
            path=path,
            sha256=digest,
            version=digest[:12],
            category=category,
        )
