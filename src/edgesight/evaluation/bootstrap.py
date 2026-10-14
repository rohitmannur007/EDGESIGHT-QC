"""Deterministic non-parametric confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    successful_resamples: int


def bootstrap_metric(
    values: Sequence[T],
    metric: Callable[[Sequence[T]], float],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    if not values:
        raise ValueError("values cannot be empty")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid resamples or confidence")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(values), size=len(values))
        value = float(metric([values[index] for index in indices]))
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        raise ValueError("metric was non-finite for every bootstrap resample")
    alpha = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=float(metric(values)),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1 - alpha)),
        confidence=confidence,
        successful_resamples=len(samples),
    )
