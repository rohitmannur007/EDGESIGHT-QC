"""MVTec AD filesystem discovery and integrity validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MVTecSample:
    path: Path
    category: str
    split: str
    defect_type: str
    is_anomaly: bool
    mask_path: Path | None = None

    @property
    def sample_id(self) -> str:
        return f"{self.category}/{self.split}/{self.defect_type}/{self.path.stem}"


def _images(path: Path) -> Iterable[Path]:
    if not path.is_dir():
        return ()
    return sorted(
        item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_mvtec(root: str | Path) -> list[MVTecSample]:
    """Discover samples from either a multi-category or single-category root."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root does not exist: {root_path}")
    categories = (
        [root_path]
        if (root_path / "train").is_dir()
        else sorted(item for item in root_path.iterdir() if item.is_dir())
    )
    samples: list[MVTecSample] = []
    for category_path in categories:
        for split in ("train", "test"):
            split_path = category_path / split
            if not split_path.is_dir():
                continue
            for defect_dir in sorted(item for item in split_path.iterdir() if item.is_dir()):
                defect_type = defect_dir.name
                for image_path in _images(defect_dir):
                    mask_path = None
                    if split == "test" and defect_type != "good":
                        candidate = (
                            category_path
                            / "ground_truth"
                            / defect_type
                            / f"{image_path.stem}_mask.png"
                        )
                        mask_path = candidate if candidate.is_file() else None
                    samples.append(
                        MVTecSample(
                            path=image_path,
                            category=category_path.name,
                            split=split,
                            defect_type=defect_type,
                            is_anomaly=defect_type != "good",
                            mask_path=mask_path,
                        )
                    )
    return samples


def validate_mvtec(root: str | Path, *, require_masks: bool = True) -> list[MVTecSample]:
    """Validate canonical MVTec invariants and return discovered samples."""

    samples = discover_mvtec(root)
    if not samples:
        raise ValueError("No MVTec images found")
    errors: list[str] = []
    categories = sorted({sample.category for sample in samples})
    for category in categories:
        category_samples = [sample for sample in samples if sample.category == category]
        if not any(sample.split == "train" for sample in category_samples):
            errors.append(f"{category}: missing training images")
        if any(sample.split == "train" and sample.is_anomaly for sample in category_samples):
            errors.append(f"{category}: train contains anomalous samples")
        if not any(sample.split == "test" for sample in category_samples):
            errors.append(f"{category}: missing test images")
    if require_masks:
        for sample in samples:
            if sample.split == "test" and sample.is_anomaly and sample.mask_path is None:
                errors.append(f"{sample.sample_id}: missing ground-truth mask")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample identifiers detected")
    if errors:
        raise ValueError("Invalid MVTec dataset:\n- " + "\n- ".join(errors))
    return samples
