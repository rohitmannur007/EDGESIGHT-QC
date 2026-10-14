#!/usr/bin/env python3
"""Evaluate precomputed detector/anomaly prediction records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edgesight.api.main import build_pipeline, load_settings
from edgesight.evaluation.pipeline import (
    evaluate_records,
    load_prediction_records,
    write_evaluation,
)


def run_models(manifest_path: Path, config_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest must contain a samples list")
    settings = load_settings(config_path)
    settings["mode"] = "real"
    pipelines: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        category = str(sample.get("category", ""))
        if category not in pipelines:
            category_settings = dict(settings)
            category_settings["category"] = category
            pipelines[category] = build_pipeline(category_settings)
        path = Path(str(sample.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"evaluation image missing: {path}")
        prediction = pipelines[category].predict(path.read_bytes(), filename=path.name)
        boxes = prediction.get("bounding_boxes") or []
        detector_score = max(
            (float(box.get("confidence", 0.0)) for box in boxes if isinstance(box, dict)),
            default=0.0,
        )
        records.append(
            {
                **sample,
                "label": sample.get("defect_type") != "good",
                "detector_score": detector_score,
                "anomaly_score": prediction.get("anomaly_score"),
                "detections": boxes,
                "decision": prediction.get("decision"),
                "inference_latency_ms": prediction.get("inference_latency_ms"),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--predictions",
        type=Path,
        help="JSON list/object or JSONL records with assignments, labels, and scores",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="Run configured real models against every custom-split manifest image",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = (
        load_prediction_records(args.predictions)
        if args.predictions
        else run_models(args.manifest, args.config)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(record, default=str) + "\n" for record in records),
        encoding="utf-8",
    )
    result = evaluate_records(
        records,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    artifacts = write_evaluation(result, args.output)
    artifacts["predictions"] = str(prediction_path)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
