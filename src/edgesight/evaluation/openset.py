"""Open-set detection metrics for unknown defect types."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .metrics import binary_metrics


def open_set_metrics(
    known_scores: Iterable[float],
    unknown_scores: Iterable[float],
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate scores where larger means more likely unknown."""

    known = np.asarray(list(known_scores), dtype=float)
    unknown = np.asarray(list(unknown_scores), dtype=float)
    if known.size == 0 or unknown.size == 0:
        raise ValueError("known and unknown score sets must both be non-empty")
    labels = np.r_[np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)]
    scores = np.r_[known, unknown]
    result = binary_metrics(labels, scores, threshold=threshold)
    result["known_acceptance_rate"] = float(np.mean(known < threshold))
    result["unknown_rejection_rate"] = float(np.mean(unknown >= threshold))
    result["threshold"] = threshold
    return result
