#!/usr/bin/env python3
"""Benchmark executable TorchScript or ONNX artifacts with real wall timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from edgesight.edge import benchmark, unsupported_benchmark
from edgesight.persistence import EdgeSightStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.input_shape.split(","))
    runtime = "unknown"
    operation: Any = None
    sample: Any = None
    reason = None
    if not args.artifact.is_file():
        reason = f"artifact not found: {args.artifact}"
    else:
        try:
            if args.artifact.suffix == ".onnx":
                runtime = "onnxruntime"
                import onnxruntime as ort  # type: ignore[import-untyped]

                session = ort.InferenceSession(
                    str(args.artifact), providers=["CPUExecutionProvider"]
                )
                input_name = session.get_inputs()[0].name
                sample = np.random.default_rng(42).normal(size=shape).astype(np.float32)

                def operation(value: Any) -> Any:
                    return session.run(None, {input_name: value})

            elif args.artifact.suffix in {".pt", ".pth"}:
                runtime = "torchscript"
                import torch

                model = torch.jit.load(str(args.artifact)).eval()  # type: ignore[no-untyped-call]
                sample = torch.randn(*shape)

                def operation(value: Any) -> Any:
                    return model(value)

            else:
                reason = f"unsupported artifact suffix: {args.artifact.suffix}"
        except Exception as exc:
            reason = str(exc)
    result = (
        unsupported_benchmark(reason or "runtime unavailable")
        if operation is None
        else benchmark(operation, sample, runs=args.runs, warmup_runs=args.warmup_runs)
    )
    record = {
        "variant": args.artifact.name,
        "runtime": runtime,
        "device": "cpu",
        "size_mb": args.artifact.stat().st_size / (1024 * 1024) if args.artifact.exists() else None,
        **result.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.database:
        store = EdgeSightStore(args.database)
        store.record_benchmark(
            variant=args.artifact.name,
            runtime=runtime,
            device="cpu",
            status=result.status,
            mean_ms=result.mean_ms if result.status == "OK" else None,
            p95_ms=result.p95_ms if result.status == "OK" else None,
            throughput_fps=(result.throughput_per_second if result.status == "OK" else None),
            size_mb=record["size_mb"],
            metadata={"reason": result.reason, "record": str(args.output)},
        )
        store.close()
    print(args.output)
    return 0 if result.status == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
