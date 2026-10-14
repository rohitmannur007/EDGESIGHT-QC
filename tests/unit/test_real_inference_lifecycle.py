from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from edgesight.decision import InspectionDecision, QualityRouter
from edgesight.inference import (
    ImagePreprocessor,
    InferenceEngine,
    InspectionPipeline,
    ModelArtifactError,
    ModelRegistry,
)
from edgesight.schemas import AnomalyResult, Detection, InferenceMode, InferenceResult


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.shape = value.shape


class FakeTorch:
    @staticmethod
    def from_numpy(value: np.ndarray) -> FakeTensor:
        return FakeTensor(value)


class FakeDetector:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict(self, image: np.ndarray) -> tuple[Detection, ...]:
        self.inputs.append(image)
        return (Detection(2, 0.91, (1, 2, 8, 9), "scratch"),)


class FakeAnomaly:
    def __init__(self) -> None:
        self.inputs: list[FakeTensor] = []

    def predict(self, tensor: FakeTensor) -> tuple[list[float], list[np.ndarray]]:
        self.inputs.append(tensor)
        return [0.2], [np.zeros((16, 16), dtype=np.float32)]


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), (20, 40, 60)).save(output, "PNG")
    return output.getvalue()


def test_real_pipeline_decodes_once_and_uses_model_specific_inputs(monkeypatch) -> None:
    opens = 0
    original_open = Image.open

    def counted_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counted_open)
    preprocessor = ImagePreprocessor(image_size=16, torch_module=FakeTorch)
    detector = FakeDetector()
    anomaly = FakeAnomaly()
    engine = InferenceEngine(
        mode="real",
        detector=detector,
        anomaly_detector=anomaly,
        preprocessor=preprocessor,
    )
    result = InspectionPipeline(engine, preprocessor=preprocessor).predict(png_bytes())

    assert opens == 1
    assert detector.inputs[0].shape == (12, 16, 3)
    assert anomaly.inputs[0].shape == (1, 3, 16, 16)
    assert result["decision"] == "KNOWN_DEFECT"
    assert result["defect_class"] == "scratch"
    assert result["synthetic"] is False


def test_router_handles_single_models_missing_models_and_failures() -> None:
    router = QualityRouter()
    detector_only = InferenceResult(
        None,
        InferenceMode.REAL,
        metadata={"configured_models": {"detector": True, "anomaly": False}},
    )
    missing = InferenceResult(None, InferenceMode.REAL)
    failed = InferenceResult(
        None,
        InferenceMode.REAL,
        anomaly=AnomalyResult(0.1),
        metadata={
            "configured_models": {"detector": False, "anomaly": True},
            "model_failures": {"anomaly": "boom"},
        },
    )

    assert router.decide(detector_only).decision is InspectionDecision.PASS
    assert router.decide(missing).decision is InspectionDecision.HUMAN_REVIEW
    assert router.decide(failed).decision is InspectionDecision.HUMAN_REVIEW


def test_registry_rejects_unsupported_api_runtime(tmp_path) -> None:
    from edgesight.inference.model_registry import UnsupportedRuntimeError

    registry = ModelRegistry(
        detector_path=tmp_path / "best.pt",
        anomaly_paths={},
        category="capsule",
        runtime="onnx",
    )
    with pytest.raises(UnsupportedRuntimeError, match="benchmark_models"):
        registry.load()


def test_registry_rejects_exported_artifact_as_pytorch_runtime(tmp_path) -> None:
    from edgesight.inference.model_registry import UnsupportedRuntimeError

    artifact = tmp_path / "detector.onnx"
    artifact.write_bytes(b"not-a-pytorch-checkpoint")
    registry = ModelRegistry(
        detector_path=artifact,
        anomaly_paths={},
        category="capsule",
        runtime="pytorch",
    )
    with pytest.raises(UnsupportedRuntimeError, match="PyTorch checkpoint"):
        registry.load()


def test_real_pipeline_applies_calibrated_or_explicit_threshold(monkeypatch) -> None:
    from edgesight.api.main import build_pipeline
    from edgesight.inference.model_registry import ModelArtifactError

    class Loaded:
        detector = None
        anomaly_detector = object()
        hashes = {"patchcore:capsule": "abc"}
        model_version = "patchcore:abc"
        runtime = "pytorch"
        anomaly_threshold = 0.37
        anomaly_threshold_source = "model_calibration"

        def __init__(self, threshold: float | None = 0.37) -> None:
            self.anomaly_threshold = threshold
            self.anomaly_threshold_source = (
                "model_calibration" if threshold is not None else "missing_calibration"
            )

    current = {"loaded": Loaded()}

    class Registry:
        @classmethod
        def from_settings(cls, settings, base_directory=None):
            return cls()

        def load(self):
            return current["loaded"]

    monkeypatch.setattr("edgesight.api.main.ModelRegistry", Registry)
    settings = {
        "mode": "real",
        "category": "capsule",
        "thresholds": {"detector": 0.7, "use_manual_anomaly_threshold": False},
    }
    pipeline = build_pipeline(settings)
    assert pipeline.anomaly_threshold == 0.37
    assert pipeline.model_info()["anomaly_threshold_source"] == "model_calibration"

    settings["thresholds"] = {
        "detector": 0.7,
        "use_manual_anomaly_threshold": True,
        "anomaly": {"capsule": 0.8},
    }
    overridden = build_pipeline(settings)
    assert overridden.anomaly_threshold == 0.8
    assert overridden.model_info()["anomaly_threshold_source"] == "manual_override"

    current["loaded"] = Loaded(None)
    settings["thresholds"] = {"detector": 0.7, "use_manual_anomaly_threshold": False}
    with pytest.raises(ModelArtifactError, match="calibration.threshold"):
        build_pipeline(settings)


def test_registry_reports_missing_artifact_without_demo_fallback(tmp_path) -> None:
    registry = ModelRegistry(
        detector_path=tmp_path / "missing.pt",
        anomaly_paths={},
        category="capsule",
    )
    with pytest.raises(ModelArtifactError, match="not found"):
        registry.load()
