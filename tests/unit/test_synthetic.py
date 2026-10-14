import pytest

np = pytest.importorskip("numpy")

from edgesight.data.synthetic import generate_synthetic_dataset  # noqa: E402
from edgesight.data.yolo import mask_to_yolo  # noqa: E402


def test_synthetic_generation_is_deterministic() -> None:
    first = generate_synthetic_dataset(8, seed=7)
    second = generate_synthetic_dataset(8, seed=7)
    assert [item.label for item in first] == [item.label for item in second]
    assert all(np.array_equal(a.image, b.image) for a, b in zip(first, second, strict=False))
    assert sum(item.label == "defect" for item in first) == 4


def test_mask_conversion_finds_separate_defects() -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[1:3, 2:6] = 1
    mask[7:9, 15:18] = 1
    labels = mask_to_yolo(mask)
    assert labels == [
        "0 0.200000 0.200000 0.200000 0.200000",
        "0 0.825000 0.800000 0.150000 0.200000",
    ]
