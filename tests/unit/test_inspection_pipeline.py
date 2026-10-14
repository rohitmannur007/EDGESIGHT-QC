from __future__ import annotations

import io

import pytest
from PIL import Image

from edgesight.inference import InferenceEngine, InspectionPipeline, InvalidImage
from edgesight.schemas import InferenceMode


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (128, 128, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_demo_pipeline_returns_complete_explicit_contract() -> None:
    pipeline = InspectionPipeline(InferenceEngine(mode=InferenceMode.DEMO))
    result = pipeline.predict(image_bytes(), filename="part.png")
    assert result["decision"] in {
        "PASS",
        "KNOWN_DEFECT",
        "UNKNOWN_DEFECT",
        "HUMAN_REVIEW",
    }
    assert result["mode"] == "demo"
    assert result["synthetic"] is True
    assert result["error"] is None
    assert result["sample_id"]
    assert result["model_version"] == "demo-deterministic-v1"


def test_pipeline_rejects_non_image_payload() -> None:
    pipeline = InspectionPipeline(InferenceEngine(mode=InferenceMode.DEMO))
    with pytest.raises(InvalidImage, match="valid image"):
        pipeline.predict(b"not an image")
