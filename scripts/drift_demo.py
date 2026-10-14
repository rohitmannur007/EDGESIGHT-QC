#!/usr/bin/env python3
"""Run a reproducible, explicitly synthetic consecutive-window drift experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edgesight.monitoring import MonitoringLoop
from edgesight.persistence import EdgeSightStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("results/drift.duckdb"))
    parser.add_argument("--output", type=Path, default=Path("results/drift_experiment.json"))
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--shift", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    baseline = {
        "brightness": rng.normal(120, 8, 500).tolist(),
        "contrast": rng.normal(35, 3, 500).tolist(),
        "blur": rng.normal(80, 5, 500).tolist(),
        "compression": rng.normal(0.7, 0.05, 500).tolist(),
        "anomaly": rng.beta(2, 12, 500).tolist(),
    }
    args.database.parent.mkdir(parents=True, exist_ok=True)
    store = EdgeSightStore(args.database)
    scenarios = {
        "clean": {},
        "brightness": {"brightness": args.shift},
        "blur": {"blur": -30.0},
        "noise": {"contrast": 18.0, "anomaly": 0.35},
        "compression": {"compression": -0.35},
    }
    experiments = {}
    for scenario, shifts in scenarios.items():
        monitor = MonitoringLoop(
            baseline,
            store=store,
            baseline_name=f"synthetic-{scenario}",
            window_size=args.window_size,
            consecutive_windows=2,
            psi_threshold=5.0,
            ks_threshold=0.4,
        )
        outcomes = []
        shifted_frames = 0
        first_alert_frame = None
        for window in range(4):
            # One clean warm-up window distinguishes abrupt onset from baseline noise.
            active_shifts = shifts if window > 0 else {}
            for index in range(args.window_size):
                shifted_frames += int(bool(active_shifts))
                outcome = monitor.observe(
                    {
                        feature: float(rng.choice(values) + active_shifts.get(feature, 0.0))
                        for feature, values in baseline.items()
                    },
                    metadata={
                        "synthetic": True,
                        "scenario": scenario,
                        "window": window,
                        "sample": index,
                    },
                )
                if outcome:
                    outcomes.append(outcome.as_dict())
                    if outcome.alerted and first_alert_frame is None:
                        first_alert_frame = shifted_frames if active_shifts else 0
        experiments[scenario] = {
            "outcomes": outcomes,
            "alert_count": sum(bool(item["alerted"]) for item in outcomes),
            "detection_delay_frames": first_alert_frame,
        }
    record = {
        "synthetic": True,
        "notice": "Synthetic drift experiment; not a production performance claim.",
        "seed": args.seed,
        "shift": args.shift,
        "window_size": args.window_size,
        "false_alerts_on_clean": experiments["clean"]["alert_count"],
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    store.record_artifact(
        "drift",
        "synthetic-brightness-shift",
        "OK",
        record,
        artifact_path=args.output,
        synthetic=True,
    )
    store.close()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
