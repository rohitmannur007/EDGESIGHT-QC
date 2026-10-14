#!/usr/bin/env python3
"""Export a caller-provided PyTorch model and write machine-readable outcomes."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from edgesight.edge import export_model, quantize_model


def load_factory(specification: str) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), attribute)()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="module:function returning a torch model")
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--output-dir", type=Path, default=Path("results/exports"))
    parser.add_argument("--formats", nargs="+", default=["torchscript", "onnx"])
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except ImportError:
        records = [{"status": "NOT_SUPPORTED", "reason": "PyTorch is not installed"}]
    else:
        model = load_factory(args.factory)
        sample = torch.randn(*(int(value) for value in args.input_shape.split(",")))
        records = []
        for format_name in args.formats:
            suffix = {"torchscript": ".pt", "onnx": ".onnx", "coreml": ".mlpackage"}.get(
                format_name, f".{format_name}"
            )
            result = export_model(
                model, sample, args.output_dir / f"model{suffix}", format=format_name
            )
            records.append(result.as_dict())
        if args.quantize:
            result = quantize_model(model)
            record = result.as_dict()
            if result.supported and result.model is not None:
                path = args.output_dir / "model-dynamic-int8.pt"
                torch.jit.trace(  # type: ignore[no-untyped-call]
                    result.model, sample.cpu()
                ).save(str(path))
                record["artifact"] = str(path)
            records.append(record)
    destination = args.output_dir / "export_records.json"
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0 if any(record.get("supported", False) for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
