"""Dataset creation, validation, and conversion helpers.

NumPy-backed utilities are loaded lazily so filesystem validation and splitting
remain available in lightweight environments.
"""

from .mvtec import MVTecSample, discover_mvtec, validate_mvtec
from .splits import DatasetSplit, mvtec_held_out_split, stratified_group_split

__all__ = [
    "DatasetSplit",
    "MVTecSample",
    "SyntheticSample",
    "discover_mvtec",
    "generate_synthetic_dataset",
    "mask_to_yolo",
    "mvtec_held_out_split",
    "prepare_from_manifest",
    "stratified_group_split",
    "validate_custom_split",
    "validate_mvtec",
]


def __getattr__(name: str):
    if name in {"prepare_from_manifest", "validate_custom_split"}:
        from . import prepare

        return getattr(prepare, name)
    if name in {"SyntheticSample", "generate_synthetic_dataset"}:
        from . import synthetic

        return getattr(synthetic, name)
    if name == "mask_to_yolo":
        from .yolo import mask_to_yolo

        return mask_to_yolo
    raise AttributeError(name)
