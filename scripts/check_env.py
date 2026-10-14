#!/usr/bin/env python3
"""Validate prerequisites without changing the environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--require-ml", action="store_true")
    parser.add_argument("--require-stream", action="store_true")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    checks.append(("Python 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0]))
    checks.append(("uv", shutil.which("uv") is not None, shutil.which("uv") or "not found"))
    checks.append(("config", args.config.is_file(), str(args.config)))

    if args.config.is_file():
        try:
            yaml.safe_load(args.config.read_text())
            checks.append(("config YAML", True, "valid"))
        except yaml.YAMLError as exc:
            checks.append(("config YAML", False, str(exc).splitlines()[0]))

    for module in ("yaml", "pydantic", "fastapi"):
        checks.append((f"module {module}", importlib.util.find_spec(module) is not None, module))
    if args.require_ml:
        for module in ("torch", "onnxruntime"):
            available = importlib.util.find_spec(module) is not None
            checks.append((f"module {module}", available, module))
        if importlib.util.find_spec("torch") is not None:
            import torch

            mps_built = bool(torch.backends.mps.is_built())
            mps_available = bool(torch.backends.mps.is_available())
            print(
                f"PyTorch: {torch.__version__}; MPS built={mps_built}, "
                f"available={mps_available}; CPU available=True"
            )
    if args.require_stream:
        stream_available = importlib.util.find_spec("confluent_kafka") is not None
        checks.append(("module confluent_kafka", stream_available, "stream extra"))
        brokers = os.getenv("EDGESIGHT_KAFKA_BROKERS", "localhost:19092")
        host, _, port = brokers.partition(":")
        try:
            with socket.create_connection((host, int(port or 9092)), timeout=1):
                reachable = True
        except OSError:
            reachable = False
        checks.append(("Kafka broker", reachable, brokers))

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    docker = shutil.which("docker")
    docker_running = False
    if docker:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        docker_running = result.returncode == 0
    print(f"INFO  Docker daemon {'running' if docker_running else 'unavailable'}")
    dataset_status = "present" if Path("data/raw/mvtec").is_dir() else "not present"
    model_extensions = {".pt", ".pth", ".onnx", ".mlpackage"}
    models_found = any(path.suffix in model_extensions for path in Path("models").rglob("*"))
    model_status = "present" if models_found else "not present"
    print(f"INFO  Real dataset {dataset_status}; model files {model_status}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
