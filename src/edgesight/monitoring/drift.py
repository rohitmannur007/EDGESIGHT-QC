"""Dependency-light distribution drift statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any


@dataclass(frozen=True)
class DriftReport:
    feature: str
    baseline_count: int
    current_count: int
    baseline_mean: float
    current_mean: float
    mean_shift_standardized: float
    ks_statistic: float
    psi: float
    drifted: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(values: Iterable[float]) -> list[float]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        raise ValueError("distribution must contain finite values")
    return cleaned


def population_stability_index(
    baseline: Iterable[float], current: Iterable[float], *, bins: int = 10
) -> float:
    reference, observed = sorted(_clean(baseline)), _clean(current)
    bins = max(2, min(bins, len(reference)))
    boundaries = [
        reference[min(len(reference) - 1, int(index * len(reference) / bins))]
        for index in range(1, bins)
    ]

    def histogram(values: list[float]) -> list[int]:
        counts = [0] * bins
        for value in values:
            index = sum(value > boundary for boundary in boundaries)
            counts[index] += 1
        return counts

    expected, actual = histogram(reference), histogram(observed)
    epsilon = 1e-6
    score = 0.0
    for expected_count, actual_count in zip(expected, actual, strict=True):
        expected_ratio = max(expected_count / len(reference), epsilon)
        actual_ratio = max(actual_count / len(observed), epsilon)
        score += (actual_ratio - expected_ratio) * math.log(actual_ratio / expected_ratio)
    return score


def _ks_statistic(left: list[float], right: list[float]) -> float:
    points = sorted(set(left + right))
    left_sorted, right_sorted = sorted(left), sorted(right)
    left_index = right_index = 0
    maximum = 0.0
    for point in points:
        while left_index < len(left_sorted) and left_sorted[left_index] <= point:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= point:
            right_index += 1
        maximum = max(maximum, abs(left_index / len(left) - right_index / len(right)))
    return maximum


def compare_distributions(
    feature: str,
    baseline: Iterable[float],
    current: Iterable[float],
    *,
    psi_threshold: float = 0.2,
    ks_threshold: float = 0.2,
) -> DriftReport:
    reference, observed = _clean(baseline), _clean(current)
    deviation = pstdev(reference)
    shift = abs(fmean(observed) - fmean(reference)) / deviation if deviation else 0.0
    psi = population_stability_index(reference, observed)
    ks = _ks_statistic(reference, observed)
    return DriftReport(
        feature=feature,
        baseline_count=len(reference),
        current_count=len(observed),
        baseline_mean=fmean(reference),
        current_mean=fmean(observed),
        mean_shift_standardized=shift,
        ks_statistic=ks,
        psi=psi,
        drifted=psi >= psi_threshold or ks >= ks_threshold,
    )


def persist_observations(store: Any, feature: str, values: Iterable[float], **metadata: Any) -> int:
    count = 0
    for value in _clean(values):
        store.record_drift(feature, value, metadata)
        count += 1
    return count


def compare_with_persisted(
    store: Any, feature: str, baseline: Iterable[float], *, limit: int = 10000
) -> DriftReport:
    return compare_distributions(feature, baseline, store.drift_values(feature, limit))
