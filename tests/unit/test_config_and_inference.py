import json

import pytest

from edgesight.config import load_config
from edgesight.inference import InferenceEngine, RealInferenceUnavailable
from edgesight.schemas import InferenceMode


def test_json_config_and_nested_override(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"router": {"anomaly_threshold": 0.8}}))
    config = load_config(path, overrides={"edge": {"input_size": [320, 320]}})
    assert config.router.anomaly_threshold == 0.8
    assert config.edge.input_size == (320, 320)


def test_demo_inference_is_explicit_and_repeatable() -> None:
    image = bytes(16 * 16 * 3)
    engine = InferenceEngine(mode=InferenceMode.DEMO)
    first = engine.predict(image)
    second = engine.predict(image)
    assert first.detections == second.detections
    assert first.anomaly.score == second.anomaly.score
    assert first.metadata["synthetic"] is True


def test_real_mode_requires_a_model() -> None:
    with pytest.raises(RealInferenceUnavailable):
        InferenceEngine(mode="real")
