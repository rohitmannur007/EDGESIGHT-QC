#!/usr/bin/env python3
"""Create and verify the requested H-site.zip source artifact."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".onnx"}


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if set(relative.parts) & EXCLUDED_PARTS:
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name in {".env", ".coverage", "H-site.zip"}:
        return False
    if relative.parts and relative.parts[0] in {"data", "models"} and path.name != ".gitkeep":
        return False
    if relative.parts and relative.parts[0] == "results":
        evidence_suffixes = {".json", ".jsonl", ".md", ".png", ".jpg", ".jpeg", ".svg", ".csv"}
        return path.name == ".gitkeep" or path.suffix.lower() in evidence_suffixes
    return True


def build(root: Path, output: Path) -> None:
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and included(path, root):
                archive.write(path, Path("H site") / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"archive CRC validation failed at {bad}")
        names = set(archive.namelist())
        required = {
            "H site/README.md",
            "H site/pyproject.toml",
            "H site/uv.lock",
            "H site/src/edgesight/api/main.py",
            "H site/tests/test_demo_e2e.py",
        }
        missing = required - names
        if missing:
            raise RuntimeError(f"archive is incomplete: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root.parent / "H-site.zip"
    build(root, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
