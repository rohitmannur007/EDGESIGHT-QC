"""Calibration and cost-sensitive operating-point selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def expected_calibration_error(
    probabilities: Iterable[float], labels: Iterable[int], *, bins: int = 10
) -> float:
    probabilities_array = np.asarray(list(probabilities), dtype=float)
    labels_array = np.asarray(list(labels), dtype=int)
    if probabilities_array.shape != labels_array.shape or probabilities_array.size == 0:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    if bins < 1 or np.any((probabilities_array < 0) | (probabilities_array > 1)):
        raise ValueError("invalid bins or probabilities")
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probabilities_array, edges[1:]), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(
                probabilities_array[selected].mean() - labels_array[selected].mean()
            )
    return float(error)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: Iterable[float], labels: Iterable[int]) -> TemperatureScaler:
        logits_array = np.asarray(list(logits), dtype=float)
        labels_array = np.asarray(list(labels), dtype=float)
        if logits_array.shape != labels_array.shape or logits_array.size == 0:
            raise ValueError("logits and labels must be non-empty and aligned")
        candidates = np.geomspace(0.05, 20.0, 400)
        losses = []
        for temperature in candidates:
            probabilities = 1 / (1 + np.exp(-np.clip(logits_array / temperature, -50, 50)))
            losses.append(
                -np.mean(
                    labels_array * np.log(probabilities + 1e-12)
                    + (1 - labels_array) * np.log(1 - probabilities + 1e-12)
                )
            )
        self.temperature = float(candidates[int(np.argmin(losses))])
        return self

    def transform(self, logits: Iterable[float]) -> np.ndarray:
        values = np.asarray(list(logits), dtype=float) / self.temperature
        return 1 / (1 + np.exp(-np.clip(values, -50, 50)))


def optimal_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    false_accept_cost: float = 10.0,
    false_reject_cost: float = 1.0,
) -> tuple[float, float]:
    """Return threshold and minimum mean cost (positive means defective)."""

    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=int)
    if score_array.shape != label_array.shape or score_array.size == 0:
        raise ValueError("scores and labels must be non-empty and aligned")
    if false_accept_cost < 0 or false_reject_cost < 0:
        raise ValueError("costs must be non-negative")
    candidates = np.unique(np.r_[np.nextafter(score_array.min(), -np.inf), score_array])
    costs = []
    for threshold in candidates:
        predictions = score_array >= threshold
        false_accepts = np.sum((predictions == 0) & (label_array == 1))
        false_rejects = np.sum((predictions == 1) & (label_array == 0))
        costs.append(
            (false_accepts * false_accept_cost + false_rejects * false_reject_cost)
            / len(score_array)
        )
    index = int(np.argmin(costs))
    return float(candidates[index]), float(costs[index])
