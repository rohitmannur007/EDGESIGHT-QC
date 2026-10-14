import pytest

from edgesight.data.mvtec import validate_mvtec


def _touch(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")


def test_mvtec_validation_resolves_masks(tmp_path) -> None:
    category = tmp_path / "widget"
    _touch(category / "train" / "good" / "001.png")
    _touch(category / "test" / "good" / "002.png")
    _touch(category / "test" / "crack" / "003.png")
    _touch(category / "ground_truth" / "crack" / "003_mask.png")
    samples = validate_mvtec(tmp_path)
    anomalous = next(sample for sample in samples if sample.is_anomaly)
    assert anomalous.mask_path.name == "003_mask.png"


def test_mvtec_validation_rejects_train_contamination(tmp_path) -> None:
    category = tmp_path / "widget"
    _touch(category / "train" / "crack" / "001.png")
    _touch(category / "test" / "good" / "002.png")
    with pytest.raises(ValueError, match="train contains anomalous"):
        validate_mvtec(tmp_path)
