#!/usr/bin/env python3
"""Optimize joint detector/anomaly thresholds from labeled JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgesight.edge import optimize_joint_thresholds
from edgesight.persistence import EdgeSightStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=Path, help="JSON list with detector_score/anomaly_score/label"
    )
    parser.add_argument("--output", type=Path, default=Path("results/cost_optimization.json"))
    parser.add_argument("--database", type=Path)
    parser.add_argument("--max-far", type=float, default=0.05)
    parser.add_argument("--missed-defect-cost", type=float, default=100.0)
    parser.add_argument("--false-reject-cost", type=float, default=5.0)
    parser.add_argument("--review-cost", type=float, default=1.0)
    parser.add_argument("--inspection-cost", type=float, default=0.02)
    parser.add_argument("--downstream-escape-cost", type=float, default=250.0)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    records = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if args.input.suffix.lower() == ".jsonl"
        else json.loads(text)
    )
    if not isinstance(records, list):
        parser.error("input must be a JSON list")
    result = optimize_joint_thresholds(
        (record["detector_score"] for record in records),
        (record["anomaly_score"] for record in records),
        (record["label"] for record in records),
        max_false_alarm_rate=args.max_far,
        missed_defect_cost=args.missed_defect_cost,
        false_reject_cost=args.false_reject_cost,
        review_cost=args.review_cost,
        inspection_cost=args.inspection_cost,
        downstream_escape_cost=args.downstream_escape_cost,
    )
    payload = result.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.database:
        store = EdgeSightStore(args.database)
        store.record_artifact(
            "cost", args.input.name, result.status, payload, artifact_path=args.output
        )
        store.close()
    print(args.output)
    return 0 if result.status == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
