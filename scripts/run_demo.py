#!/usr/bin/env python3
"""Generate a deterministic, explicitly synthetic dataset using the package API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from edgesight.data.synthetic import generate_synthetic_dataset, save_synthetic_dataset
from edgesight.inference import InferenceEngine, InspectionPipeline
from edgesight.monitoring.drift import compare_distributions
from edgesight.persistence import EdgeSightStore
from edgesight.schemas import InferenceMode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("SYNTHETIC DEMO: outputs are illustrative and are not benchmark results.", flush=True)
    config = yaml.safe_load(args.config.read_text())
    demo = config["demo"]
    samples = generate_synthetic_dataset(
        int(demo["samples"]),
        anomaly_fraction=float(demo["defect_probability"]),
        seed=int(config.get("seed", config.get("project", {}).get("seed", 42))),
    )
    save_synthetic_dataset(samples, args.output_dir / "samples")
    images_dir = args.output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    visualizations_dir = args.output_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)
    store = EdgeSightStore(args.output_dir / "demo.duckdb")
    pipeline = InspectionPipeline(
        InferenceEngine(mode=InferenceMode.DEMO),
        anomaly_threshold_source="demo_default",
    )
    predictions = []
    evaluation_records = []
    brightness = []
    validation_count = max(4, len(samples) // 4)
    for index, sample in enumerate(samples):
        path = images_dir / f"{sample.sample_id}.png"
        Image.fromarray(sample.image).save(path)
        payload = path.read_bytes()
        prediction = pipeline.predict(payload, filename=path.name)
        store.save_result(status="complete", prediction=prediction, source_name=path.name)
        predictions.append(prediction)
        assignment = (
            "validation"
            if index < validation_count
            else "good_test"
            if sample.label == "good"
            else "unseen_test"
            if index % 2
            else "known_test"
        )
        evaluation_records.append(
            {
                "sample_id": sample.sample_id,
                "category": "synthetic-demo",
                "assignment": assignment,
                "label": sample.label == "defect",
                "defect_type": (
                    "good"
                    if sample.label == "good"
                    else "synthetic_unseen"
                    if assignment == "unseen_test"
                    else "synthetic_known"
                ),
                "detector_score": prediction["detector_confidence"],
                "anomaly_score": prediction["anomaly_score"],
                "detections": prediction["bounding_boxes"],
                "synthetic": True,
            }
        )
        brightness.append(float(np.mean(sample.image)))
        if len(predictions) <= 10:
            annotated = Image.fromarray(sample.image)
            draw = ImageDraw.Draw(annotated)
            for box in prediction["bounding_boxes"]:
                draw.rectangle(box["xyxy"], outline="red", width=2)
                draw.text(
                    (box["xyxy"][0], max(0, box["xyxy"][1] - 10)),
                    f"{box['label']} {box['confidence']:.2f}",
                    fill="red",
                )
            draw.text((4, 4), f"SYNTHETIC {prediction['decision']}", fill="yellow")
            annotated.save(visualizations_dir / f"{sample.sample_id}-annotated.png")
    shifted = [min(255.0, value + 35.0) for value in brightness]
    drift_report = compare_distributions("brightness", brightness, shifted).as_dict()
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "evaluation_predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in evaluation_records),
        encoding="utf-8",
    )
    (args.output_dir / "drift_report.json").write_text(
        json.dumps(drift_report, indent=2) + "\n", encoding="utf-8"
    )
    store.close()
    manifest = {
        "synthetic": True,
        "benchmark": False,
        "sample_count": len(samples),
        "labels": {
            label: sum(sample.label == label for sample in samples) for label in ("good", "defect")
        },
        "notice": "Illustrative synthetic data; not a production metric.",
        "database": "demo.duckdb",
        "predictions": "predictions.json",
        "evaluation_predictions": "evaluation_predictions.jsonl",
        "drift_report": "drift_report.json",
        "visualizations": "visualizations/",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
