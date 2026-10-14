"""Uvicorn entry point with explicit demo and production model loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from edgesight.api.app import create_app
from edgesight.inference import (
    ImagePreprocessor,
    InferenceEngine,
    InspectionPipeline,
    ModelRegistry,
)
from edgesight.inference.model_registry import ModelArtifactError
from edgesight.schemas import InferenceMode

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(
        path or os.getenv("EDGESIGHT_CONFIG_PATH", PROJECT_ROOT / "configs" / "app.yaml")
    )
    settings: dict[str, Any] = {}
    if config_path.is_file():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load API settings") from exc
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"API settings must be a mapping: {config_path}")
        settings.update(loaded)
    elif path is not None or "EDGESIGHT_CONFIG_PATH" in os.environ:
        raise FileNotFoundError(f"API settings file not found: {config_path}")

    settings["mode"] = os.getenv("EDGESIGHT_MODE", str(settings.get("mode", "demo")))
    settings["category"] = os.getenv("EDGESIGHT_CATEGORY", str(settings.get("category", "default")))
    settings["device"] = os.getenv("EDGESIGHT_DEVICE", str(settings.get("device", "cpu")))
    if settings["device"] == "auto":
        try:
            import torch

            settings["device"] = "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            settings["device"] = "cpu"
    models = dict(settings.get("models") or {})
    if detector_path := os.getenv("EDGESIGHT_DETECTOR_PATH"):
        models["detector_path"] = detector_path
    if anomaly_path := os.getenv("EDGESIGHT_ANOMALY_MODEL_PATH"):
        anomaly_models = dict(models.get("anomaly_models") or {})
        anomaly_models[settings["category"]] = anomaly_path
        models["anomaly_models"] = anomaly_models
    settings["models"] = models
    settings["_base_directory"] = str(PROJECT_ROOT)
    return settings


def _manual_anomaly_threshold(settings: dict[str, Any], category: str) -> float | None:
    thresholds = settings.get("thresholds") or {}
    manual = isinstance(thresholds, dict) and thresholds.get("use_manual_anomaly_threshold", False)
    if not manual:
        return None
    anomaly_values = thresholds.get("anomaly", {})
    if isinstance(anomaly_values, dict):
        if category not in anomaly_values:
            raise ModelArtifactError(
                "thresholds.use_manual_anomaly_threshold is true, but no anomaly "
                f"threshold is configured for category {category!r}"
            )
        return float(anomaly_values[category])
    return float(anomaly_values)


def build_pipeline(settings: dict[str, Any] | None = None) -> InspectionPipeline:
    settings = settings or load_settings()
    mode = InferenceMode(settings.get("mode", "demo"))
    category = str(settings.get("category", "default"))
    thresholds = settings.get("thresholds") or {}
    detector_threshold = float(thresholds.get("detector", 0.7))
    manual_anomaly_threshold = _manual_anomaly_threshold(settings, category)
    preprocessor = ImagePreprocessor(
        image_size=int(settings.get("image_size", 256)),
        max_pixels=int(settings.get("max_pixels", 25_000_000)),
    )
    if mode is InferenceMode.DEMO:
        anomaly_threshold = (
            manual_anomaly_threshold if manual_anomaly_threshold is not None else 0.5
        )
        return InspectionPipeline(
            InferenceEngine(
                mode=mode,
                preprocessor=preprocessor,
                anomaly_threshold=anomaly_threshold,
            ),
            detector_threshold=detector_threshold,
            anomaly_threshold=anomaly_threshold,
            anomaly_threshold_source=(
                "manual_override" if manual_anomaly_threshold is not None else "demo_default"
            ),
            category=category,
            preprocessor=preprocessor,
            model_version=str(settings.get("model_version", "demo-deterministic-v1")),
            artifact_directory=settings.get("artifact_directory"),
        )

    loaded = ModelRegistry.from_settings(
        settings, base_directory=settings.get("_base_directory", PROJECT_ROOT)
    ).load()
    if manual_anomaly_threshold is not None:
        anomaly_threshold = manual_anomaly_threshold
        anomaly_source = "manual_override"
    elif loaded.anomaly_detector is not None:
        if loaded.anomaly_threshold is None:
            raise ModelArtifactError(
                "PatchCore artifact has no calibration.threshold. Refit it with "
                "normal validation images or set thresholds.use_manual_anomaly_threshold."
            )
        anomaly_threshold = loaded.anomaly_threshold
        anomaly_source = loaded.anomaly_threshold_source
    else:
        anomaly_threshold = 0.5
        anomaly_source = "unconfigured"
    metadata = {
        "model_hashes": loaded.hashes,
        "model_version": loaded.model_version,
        "runtime": loaded.runtime,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_threshold_source": anomaly_source,
    }
    engine = InferenceEngine(
        mode=mode,
        detector=loaded.detector,
        anomaly_detector=loaded.anomaly_detector,
        anomaly_threshold=anomaly_threshold,
        preprocessor=preprocessor,
        model_metadata=metadata,
    )
    return InspectionPipeline(
        engine,
        detector_threshold=detector_threshold,
        anomaly_threshold=anomaly_threshold,
        anomaly_threshold_source=anomaly_source,
        category=category,
        preprocessor=preprocessor,
        model_version=loaded.model_version,
        model_hashes=loaded.hashes,
        runtime=loaded.runtime,
        artifact_directory=settings.get("artifact_directory"),
    )


_settings = load_settings()
_security = _settings.get("security") or {}
_api_key_env = str(_security.get("api_key_env", "EDGESIGHT_API_KEY"))
app = create_app(
    build_pipeline(_settings),
    max_upload_bytes=int(_settings.get("max_upload_bytes", 10 * 1024 * 1024)),
    max_batch_size=int(_settings.get("max_batch_size", 100)),
    require_api_key=bool(_security.get("require_api_key", False)),
    api_key=os.getenv(_api_key_env),
)
