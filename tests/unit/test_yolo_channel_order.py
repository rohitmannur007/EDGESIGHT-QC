import numpy as np
from PIL import Image

from edgesight.detector.yolo import YoloDetector


def test_numpy_rgb_is_converted_to_bgr_without_mutating_input() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 9
    rgb[..., 1] = 20
    rgb[..., 2] = 40

    converted = YoloDetector._model_input(rgb)

    assert converted.shape == rgb.shape
    assert converted[0, 0].tolist() == [40, 20, 9]
    assert rgb[0, 0].tolist() == [9, 20, 40]
    assert converted.flags["C_CONTIGUOUS"]


def test_non_numpy_inputs_keep_their_channel_convention() -> None:
    image = Image.new("RGB", (2, 2), (9, 20, 40))
    assert YoloDetector._model_input(image) is image


def test_predict_passes_bgr_numpy_to_ultralytics(tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"unused-by-fake-model")
    detector = YoloDetector(weights)
    captured: dict[str, np.ndarray] = {}

    class FakeYolo:
        def predict(self, source, **kwargs):
            captured["source"] = source
            return []

    detector._model = FakeYolo()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 7
    rgb[..., 2] = 15
    assert detector.predict(rgb) == ()
    assert captured["source"][0, 0].tolist() == [15, 0, 7]
