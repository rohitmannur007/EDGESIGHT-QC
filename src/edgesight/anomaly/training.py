"""Category-specific PatchCore training with explicit local weight policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .patchcore import PatchCore, _torch

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def _batches(paths: list[Path], *, size: int, batch_size: int) -> list[Any]:
    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("anomaly training requires Pillow and torchvision") from exc
    torch = _torch()
    preprocess = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    tensors = [preprocess(Image.open(path).convert("RGB")) for path in paths]
    return [
        torch.stack(tensors[start : start + batch_size])
        for start in range(0, len(tensors), batch_size)
    ]


def _backbone(config: dict[str, Any]) -> Any | None:
    if not config.get("pretrained_backbone", False):
        return None
    if config.get("pretrained_weights_license_reviewed") is not True:
        raise RuntimeError("pretrained weights license must be reviewed and acknowledged")
    weights_path = Path(str(config.get("backbone_weights", "")))
    if not weights_path.is_file():
        raise FileNotFoundError(
            "pretrained_backbone requires a local anomaly.backbone_weights artifact"
        )
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise RuntimeError("anomaly training requires torchvision") from exc
    factory = getattr(models, str(config.get("backbone", "resnet18")), None)
    if factory is None:
        raise ValueError(f"unsupported backbone: {config.get('backbone')}")
    model = factory(weights=None)
    torch = _torch()
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    return model


def train_category_models(
    normal_root: str | Path,
    output_root: str | Path,
    *,
    config: dict[str, Any],
    seed: int = 42,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one artifact per category using train normals and validation calibration."""

    torch = _torch()
    root = Path(normal_root)
    output = Path(output_root)
    categories = sorted(path.parent.name for path in root.glob("*/normal_train") if path.is_dir())
    if not categories:
        raise FileNotFoundError(f"no category/normal_train directories under {root}")
    requested_device = str(config.get("device", "cpu"))
    actual_device = requested_device
    if (
        requested_device == "mps"
        and not torch.backends.mps.is_available()
        or requested_device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        actual_device = "cpu"
    size = int(config.get("image_size", 256))
    batch_size = int(config.get("batch_size", 4))
    maximum_images = int(config.get("maximum_fit_images", 200))
    metadata: dict[str, Any] = {
        "requested_device": requested_device,
        "actual_device": actual_device,
        "device_fallback": requested_device != actual_device,
        "categories": {},
    }
    for category in categories:
        artifact = output / category / "patchcore.pt"
        if resume and artifact.is_file():
            metadata["categories"][category] = {
                "status": "resumed_completed",
                "artifact": str(artifact),
            }
            continue
        train_paths = _images(root / category / "normal_train")[:maximum_images]
        validation_paths = _images(root / category / "normal_validation")
        if not train_paths:
            raise FileNotFoundError(f"{category}: no normal training images")
        preprocess_metadata = {
            "resize": [size, size],
            "color": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        model = PatchCore(
            backbone=_backbone(config),
            backbone_name=str(config.get("backbone", "resnet18")),
            layers=tuple(config.get("layers", ("layer2", "layer3"))),
            memory_samples=int(config.get("maximum_bank_size", 10_000)),
            candidate_reservoir_size=int(
                config.get(
                    "candidate_reservoir_size",
                    int(config.get("maximum_bank_size", 10_000)) * 4,
                )
            ),
            coreset_strategy=str(config.get("coreset_strategy", "greedy")),
            distance_chunk_size=int(config.get("distance_chunk_size", 512)),
            device=actual_device,
            seed=seed,
            pretrained=False,
            category=category,
            preprocess=preprocess_metadata,
            artifact_references={
                "normal_root": str(root.resolve()),
                "backbone_weights": str(config.get("backbone_weights", "")),
            },
        ).fit(
            _batches(train_paths, size=size, batch_size=batch_size),
            calibration_batches=(
                _batches(validation_paths, size=size, batch_size=batch_size)
                if validation_paths
                else None
            ),
            calibration_quantile=float(config.get("calibration_quantile", 0.99)),
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        model.save(artifact)
        metadata["categories"][category] = {
            "status": "trained",
            "artifact": str(artifact),
            "train_images": len(train_paths),
            "calibration_images": len(validation_paths),
            "calibration": model.calibration,
        }
        model.close()
        snapshot = output / "snapshots" / f"{category}.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(
            json.dumps(metadata["categories"][category], indent=2) + "\n",
            encoding="utf-8",
        )
    return metadata
