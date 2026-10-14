#!/usr/bin/env python3
"""Validate a user-supplied MVTec AD tree and create the custom split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from edgesight.data.mvtec import MVTecSample, validate_mvtec
from edgesight.data.prepare import prepare_from_manifest

HELD_OUT = {"capsule": "squeeze", "hazelnut": "print", "tile": "rough"}


def _stable(samples: list[MVTecSample], seed: int) -> list[MVTecSample]:
    return sorted(
        samples,
        key=lambda sample: hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).hexdigest(),
    )


def _record(sample: MVTecSample, assignment: str, digest: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "path": str(sample.path.resolve()),
        "mask_path": str(sample.mask_path.resolve()) if sample.mask_path else None,
        "category": sample.category,
        "defect_type": sample.defect_type,
        "source_split": sample.split,
        "assignment": assignment,
        "sha256": digest,
        "mask_sha256": hashlib.sha256(sample.mask_path.read_bytes()).hexdigest()
        if sample.mask_path
        else None,
    }


def validate_and_manifest(source: Path, *, seed: int = 42) -> dict[str, Any]:
    samples = [sample for sample in validate_mvtec(source) if sample.category in HELD_OUT]
    categories = {sample.category for sample in samples}
    missing_categories = set(HELD_OUT) - categories
    if missing_categories:
        raise ValueError(f"missing required categories: {sorted(missing_categories)}")
    defect_names = {(sample.category, sample.defect_type) for sample in samples}
    missing_heldout = [
        f"{category}/{defect}"
        for category, defect in HELD_OUT.items()
        if (category, defect) not in defect_names
    ]
    if missing_heldout:
        raise ValueError(f"held-out defect types not found: {missing_heldout}")

    hashes: dict[str, str] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    seen_hashes: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                image.verify()
            with Image.open(sample.path) as image:
                dimensions[sample.sample_id] = image.size
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"corrupt image: {sample.path}") from exc
        digest = hashlib.sha256(sample.path.read_bytes()).hexdigest()
        hashes[sample.sample_id] = digest
        if digest in seen_hashes:
            duplicates.append((seen_hashes[digest], sample.sample_id))
        seen_hashes[digest] = sample.sample_id
        if sample.mask_path:
            with Image.open(sample.mask_path) as mask:
                if mask.size != dimensions[sample.sample_id]:
                    raise ValueError(f"mask dimensions differ for {sample.sample_id}")
    if duplicates:
        raise ValueError(f"duplicate image content detected: {duplicates[:10]}")

    records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[MVTecSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.category, sample.defect_type)].append(sample)
    for (category, defect), members in sorted(grouped.items()):
        ordered = _stable(members, seed)
        if defect == HELD_OUT[category]:
            assignments = ["unseen_test"] * len(ordered)
        elif defect == "good":
            assignments = ["train" if item.split == "train" else "good_test" for item in ordered]
            train_indices = [index for index, item in enumerate(ordered) if item.split == "train"]
            validation_count = (
                min(len(train_indices) - 1, max(1, round(len(train_indices) * 0.2)))
                if len(train_indices) > 1
                else 0
            )
            for index in train_indices[:validation_count]:
                assignments[index] = "validation"
        else:
            train_end = round(len(ordered) * 0.6)
            validation_end = train_end + round(len(ordered) * 0.2)
            assignments = [
                "known_train"
                if index < train_end
                else "known_validation"
                if index < validation_end
                else "known_test"
                for index in range(len(ordered))
            ]
        records.extend(
            _record(sample, assignment, hashes[sample.sample_id])
            for sample, assignment in zip(ordered, assignments, strict=True)
        )
    return {
        "protocol": "custom-open-set-v1-not-official-mvtec",
        "seed": seed,
        "held_out": HELD_OUT,
        "counts": dict(Counter(record["assignment"] for record in records)),
        "samples": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/splits/splits.json"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--minimum-component-area", type=int, default=16)
    args = parser.parse_args()
    manifest = validate_and_manifest(args.source)
    print(json.dumps({"counts": manifest["counts"], "held_out": manifest["held_out"]}, indent=2))
    if args.prepare:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stats = prepare_from_manifest(
            args.output,
            args.processed_root,
            minimum_component_area=args.minimum_component_area,
        )
        print(json.dumps({"manifest": str(args.output), "prepared": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
