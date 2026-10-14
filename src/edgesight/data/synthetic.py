"""Deterministic synthetic quality-control images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SyntheticSample:
    image: np.ndarray
    mask: np.ndarray
    label: str
    sample_id: str


def generate_synthetic_dataset(
    count: int,
    *,
    image_size: tuple[int, int] = (128, 128),
    anomaly_fraction: float = 0.5,
    seed: int = 42,
) -> list[SyntheticSample]:
    """Generate repeatable textured parts with optional rectangular defects."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if not 0.0 <= anomaly_fraction <= 1.0:
        raise ValueError("anomaly_fraction must be in [0, 1]")
    height, width = image_size
    if min(height, width) < 16:
        raise ValueError("each image dimension must be at least 16")

    rng = np.random.default_rng(seed)
    anomalous = int(round(count * anomaly_fraction))
    labels = np.array([True] * anomalous + [False] * (count - anomalous))
    rng.shuffle(labels)
    samples: list[SyntheticSample] = []
    yy, xx = np.mgrid[:height, :width]
    base_pattern = 110 + 18 * np.sin(xx / 7.0) + 12 * np.cos(yy / 11.0)
    for index, has_anomaly in enumerate(labels):
        noise = rng.normal(0, 5, size=(height, width))
        grayscale = np.clip(base_pattern + noise, 0, 255).astype(np.uint8)
        image = np.repeat(grayscale[..., None], 3, axis=2)
        mask = np.zeros((height, width), dtype=np.uint8)
        if has_anomaly:
            defect_h = int(rng.integers(max(3, height // 16), max(4, height // 5)))
            defect_w = int(rng.integers(max(3, width // 16), max(4, width // 5)))
            top = int(rng.integers(0, height - defect_h + 1))
            left = int(rng.integers(0, width - defect_w + 1))
            mask[top : top + defect_h, left : left + defect_w] = 1
            tint = rng.integers(40, 220, size=3, dtype=np.uint8)
            image[mask.astype(bool)] = tint
        samples.append(
            SyntheticSample(
                image=image,
                mask=mask,
                label="defect" if has_anomaly else "good",
                sample_id=f"synthetic-{seed}-{index:06d}",
            )
        )
    return samples


def save_synthetic_dataset(
    samples: Sequence[SyntheticSample], destination: str | Path
) -> None:
    """Save samples without requiring image codecs, using compressed NumPy files."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        np.savez_compressed(
            root / f"{sample.sample_id}.npz",
            image=sample.image,
            mask=sample.mask,
            label=np.asarray(sample.label),
        )
