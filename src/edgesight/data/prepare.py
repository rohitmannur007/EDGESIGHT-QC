"""Materialize leakage-safe detector and anomaly datasets from a split manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .yolo import mask_to_yolo

HELDOUT_ASSIGNMENTS = {"unseen_test"}
CALIBRATION_ASSIGNMENTS = {"validation", "known_validation"}
TRAIN_ASSIGNMENTS = {"train", "known_train"}
EVALUATION_ASSIGNMENTS = {"good_test", "known_test", "unseen_test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list):
        raise ValueError("split manifest must contain a samples list")
    return value


def validate_custom_split(manifest: dict[str, Any]) -> None:
    """Reject source changes, duplicate content, and held-out leakage."""

    held_out = manifest.get("held_out", {})
    if not isinstance(held_out, dict):
        raise ValueError("manifest held_out must be an object")
    seen_ids: set[str] = set()
    digest_assignments: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for record in manifest["samples"]:
        sample_id = str(record.get("sample_id", ""))
        assignment = str(record.get("assignment", ""))
        category = str(record.get("category", ""))
        defect = str(record.get("defect_type", ""))
        path = Path(str(record.get("path", "")))
        if not sample_id or sample_id in seen_ids:
            errors.append(f"duplicate or empty sample_id: {sample_id!r}")
        seen_ids.add(sample_id)
        if not path.is_file():
            errors.append(f"{sample_id}: source image missing")
            continue
        actual = _sha256(path)
        if actual != record.get("sha256"):
            errors.append(f"{sample_id}: image hash differs from validated manifest")
        if record.get("mask_path"):
            mask_path = Path(str(record["mask_path"]))
            if not mask_path.is_file():
                errors.append(f"{sample_id}: mask missing")
            elif record.get("mask_sha256") and _sha256(mask_path) != record["mask_sha256"]:
                errors.append(f"{sample_id}: mask hash differs from validated manifest")
        digest_assignments[actual].add(assignment)
        expected_heldout = held_out.get(category)
        if defect == expected_heldout and assignment != "unseen_test":
            errors.append(f"{sample_id}: held-out defect assigned to {assignment}")
        if assignment == "unseen_test" and defect != expected_heldout:
            errors.append(f"{sample_id}: non-held-out defect assigned unseen_test")
        if defect == "good" and assignment not in {"train", "validation", "good_test"}:
            errors.append(f"{sample_id}: normal sample has invalid assignment {assignment}")
        if defect != "good" and assignment in {"train", "validation"}:
            errors.append(f"{sample_id}: anomalous sample assigned as normal")
    for digest, assignments in digest_assignments.items():
        if len(assignments) > 1:
            errors.append(f"content {digest[:12]} appears across assignments {sorted(assignments)}")
    if errors:
        raise ValueError("Invalid custom split:\n- " + "\n- ".join(errors))


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _boxes(rows: list[str], width: int, height: int) -> list[tuple[int, int, int, int]]:
    result = []
    for row in rows:
        _, cx, cy, box_width, box_height = row.split()
        x1 = int((float(cx) - float(box_width) / 2) * width)
        y1 = int((float(cy) - float(box_height) / 2) * height)
        x2 = int((float(cx) + float(box_width) / 2) * width)
        y2 = int((float(cy) + float(box_height) / 2) * height)
        result.append((x1, y1, x2, y2))
    return result


def prepare_from_manifest(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    minimum_component_area: int = 16,
    visualization_limit: int = 12,
) -> dict[str, Any]:
    """Create YOLO and per-category PatchCore trees and return preparation stats."""

    manifest_file = Path(manifest_path)
    output = Path(output_root)
    manifest = _load_manifest(manifest_file)
    validate_custom_split(manifest)
    records: list[dict[str, Any]] = manifest["samples"]
    classes = sorted(
        {
            f"{record['category']}:{record['defect_type']}"
            for record in records
            if record["defect_type"] != "good"
            and record["assignment"] not in HELDOUT_ASSIGNMENTS
        }
    )
    class_ids = {name: index for index, name in enumerate(classes)}
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    boxes_count = 0
    visualized = 0
    materialized: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: item["sample_id"]):
        source = Path(record["path"])
        assignment = record["assignment"]
        category = record["category"]
        defect = record["defect_type"]
        destination_paths: list[Path] = []

        if assignment in {"train", "known_train", "validation", "known_validation"}:
            yolo_split = "train" if assignment in TRAIN_ASSIGNMENTS else "val"
            image_name = f"{category}__{defect}__{source.name}"
            image_destination = output / "yolo" / "images" / yolo_split / image_name
            label_destination = (
                output
                / "yolo"
                / "labels"
                / yolo_split
                / f"{Path(image_name).stem}.txt"
            )
            _copy(source, image_destination)
            rows: list[str] = []
            if defect != "good":
                mask_path = Path(record["mask_path"]) if record.get("mask_path") else None
                if mask_path is None or not mask_path.is_file():
                    raise ValueError(f"{record['sample_id']}: known anomaly has no mask")
                with Image.open(mask_path) as mask:
                    rows = mask_to_yolo(
                        np.asarray(mask), class_id=class_ids[f"{category}:{defect}"],
                        min_area=minimum_component_area,
                    )
                if not rows:
                    raise ValueError(f"{record['sample_id']}: mask produced no bounding boxes")
                boxes_count += len(rows)
                if visualized < visualization_limit:
                    with Image.open(source) as image:
                        preview = image.convert("RGB")
                        draw = ImageDraw.Draw(preview)
                        for box in _boxes(rows, *preview.size):
                            draw.rectangle(box, outline=(255, 32, 32), width=3)
                        preview_path = output / "visualizations" / f"{Path(image_name).stem}.jpg"
                        preview_path.parent.mkdir(parents=True, exist_ok=True)
                        preview.save(preview_path, quality=90)
                        visualized += 1
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            destination_paths.extend((image_destination, label_destination))
            counts[f"yolo_{yolo_split}_images"] += 1
            if not rows:
                counts[f"yolo_{yolo_split}_negatives"] += 1

        if defect == "good" and assignment in {"train", "validation"}:
            anomaly_split = "normal_train" if assignment == "train" else "normal_validation"
            anomaly_destination = output / "anomaly" / category / anomaly_split / source.name
            _copy(source, anomaly_destination)
            destination_paths.append(anomaly_destination)
            counts[f"{category}_{anomaly_split}"] += 1

        materialized.append(
            {
                **record,
                "mask_sha256": _sha256(Path(record["mask_path"]))
                if record.get("mask_path")
                else None,
                "materialized": [
                    {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                    }
                    for path in destination_paths
                ],
            }
        )

    yolo_root = (output / "yolo").resolve()
    (output / "yolo").mkdir(parents=True, exist_ok=True)
    (output / "yolo" / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for name, index in class_ids.items()},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats = {
        "protocol": manifest.get("protocol"),
        "seed": manifest.get("seed"),
        "class_ids": class_ids,
        "counts": dict(sorted(counts.items())),
        "bounding_boxes": boxes_count,
        "visualizations": visualized,
        "held_out": manifest.get("held_out"),
        "leakage_checks": {
            "disjoint_content": True,
            "heldout_excluded_from_training": True,
            "heldout_excluded_from_calibration": True,
        },
    }
    (output / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prepared_manifest = {
        **{key: value for key, value in manifest.items() if key != "samples"},
        "class_ids": class_ids,
        "samples": materialized,
    }
    (output / "manifest.json").write_text(
        json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats
