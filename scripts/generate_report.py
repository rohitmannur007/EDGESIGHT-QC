#!/usr/bin/env python3
"""Create an evidence index from existing result artifacts; never synthesize metrics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def flatten_metrics(value: Any, prefix: str = "") -> list[tuple[str, int | float | str]]:
    rows: list[tuple[str, int | float | str]] = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_metrics(child, name))
    elif isinstance(value, int | float | str) and not isinstance(value, bool):
        rows.append((prefix, value))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/report.md"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_files = sorted(p for p in args.results_dir.rglob("*.json") if p != args.output)
    lines = [
        "# EdgeSight-QC Evaluation Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "> This report only transcribes existing artifacts. Missing values are not estimated.",
        "",
    ]
    if not json_files:
        lines += ["## Status", "", "No JSON evaluation artifacts were found.", ""]
    for path in json_files:
        lines += [f"## `{path.relative_to(args.results_dir)}`", ""]
        try:
            rows = flatten_metrics(json.loads(path.read_text()))
            lines += [f"- `{name}`: {value}" for name, value in rows] or ["No scalar fields found."]
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"Could not parse artifact: `{exc}`")
        lines.append("")
    args.output.write_text("\n".join(lines))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
