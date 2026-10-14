#!/usr/bin/env python3
"""Train the detector and category PatchCore artifacts with explicit prerequisites."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training configuration not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def _acknowledged(config: dict[str, Any]) -> bool:
    return config.get("dataset", {}).get("license_acknowledged") is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--anomaly-config", type=Path, default=Path("configs/anomaly.yaml"))
    parser.add_argument(
        "--family", action="append", choices=("detector", "anomaly"), dest="families"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metadata", type=Path, default=Path("models/training-metadata.json"))
    args = parser.parse_args()

    shared = _load(args.config) if args.config else {}
    families = args.families or shared.get("training", {}).get(
        "model_families", ["detector", "anomaly"]
    )
    if set(families) - {"detector", "anomaly"}:
        parser.error("only detector and anomaly model families are supported")
    metadata: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "resume": args.resume,
        "families": {},
    }
    for family in families:
        config_path = args.detector_config if family == "detector" else args.anomaly_config
        config = _load(config_path)
        if shared:
            config["seed"] = shared.get("seed", config.get("seed", 42))
            config["dataset"] = {
                **config.get("dataset", {}),
                **shared.get("dataset", {}),
            }
        if not _acknowledged(config):
            print(
                f"Refusing {family} training: acknowledge the dataset license in "
                f"{args.config or config_path}.",
                file=sys.stderr,
            )
            return 2
        if family == "detector":
            from edgesight.detector.yolo import train_yolo

            values = config.get("detector", {})
            if values.get("pretrained_weights_license_reviewed") is not True:
                raise RuntimeError("detector.pretrained_weights_license_reviewed must be true")
            data = Path(values.get("data", "data/processed/yolo/data.yaml"))
            base_model = Path(values.get("base_model", "models/detector/yolo11n.pt"))
            output = Path(values.get("output_dir", "models/detector/runs"))
            if not data.is_file():
                raise FileNotFoundError(f"prepared YOLO data config missing: {data}")
            resume_weights = sorted(output.rglob("last.pt"), key=lambda path: path.stat().st_mtime)
            selected_weights = resume_weights[-1] if args.resume and resume_weights else base_model
            if not selected_weights.is_file():
                raise FileNotFoundError(
                    f"local detector weights missing: {selected_weights}; no implicit download"
                )
            requested_device = str(values.get("device", "cpu"))
            actual_device = requested_device
            try:
                import torch

                if (
                    requested_device == "mps"
                    and not torch.backends.mps.is_available()
                    or requested_device.startswith("cuda")
                    and not torch.cuda.is_available()
                ):
                    actual_device = "cpu"
            except ImportError:
                if requested_device != "cpu":
                    actual_device = "cpu"
            result = train_yolo(
                data,
                base_model=selected_weights,
                output_directory=output,
                epochs=int(values.get("epochs", 50)),
                image_size=int(values.get("image_size", 640)),
                seed=int(config.get("seed", 42)),
                batch=int(values.get("batch_size", 4)),
                device=actual_device,
                resume=bool(args.resume and resume_weights),
            )
            save_dir = Path(getattr(result, "save_dir", output))
            candidates = list(save_dir.rglob("best.pt")) or list(output.rglob("best.pt"))
            if not candidates:
                raise FileNotFoundError(f"YOLO training finished without best.pt under {save_dir}")
            best = max(candidates, key=lambda path: path.stat().st_mtime)
            promoted = Path(values.get("promoted_path", "models/detector/best.pt"))
            promoted.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, promoted)
            metadata["families"]["detector"] = {
                "status": "trained",
                "data": str(data),
                "weights": str(selected_weights),
                "resumed": bool(args.resume and resume_weights),
                "requested_device": requested_device,
                "actual_device": actual_device,
                "device_fallback": requested_device != actual_device,
                "save_dir": str(save_dir),
                "best_checkpoint": str(best),
                "promoted_path": str(promoted),
            }
        else:
            from edgesight.anomaly.training import train_category_models

            values = config.get("anomaly", {})
            normal_root = Path(values.get("normal_image_dir", "data/processed/anomaly"))
            # Older configs pointed at a shared normal_train directory.
            if normal_root.name == "normal_train":
                normal_root = normal_root.parent
            output_value = Path(values.get("output", "models/anomaly/patchcore.pt"))
            output = output_value.parent if output_value.suffix else output_value
            metadata["families"]["anomaly"] = train_category_models(
                normal_root,
                output,
                config=values,
                seed=int(config.get("seed", 42)),
                resume=args.resume,
            )
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
