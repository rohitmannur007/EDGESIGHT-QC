import math

import pytest

pytest.importorskip("numpy")

from edgesight.decision.calibration import (
    expected_calibration_error,
    optimal_threshold,
)
from edgesight.evaluation import binary_metrics, bootstrap_metric, open_set_metrics


def test_binary_metrics_and_open_set() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    opened = open_set_metrics([0.1, 0.2], [0.8, 0.9], threshold=0.5)
    assert opened["unknown_rejection_rate"] == 1.0


def test_cost_and_calibration_helpers() -> None:
    threshold, cost = optimal_threshold([0.1, 0.3, 0.8], [0, 0, 1])
    assert math.isclose(threshold, 0.8)
    assert cost == 0.0
    assert expected_calibration_error([0.1, 0.9], [0, 1], bins=2) == pytest.approx(0.1)


def test_bootstrap_is_seeded() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    def metric(sample: list[float]) -> float:
        return sum(sample) / len(sample)

    assert bootstrap_metric(values, metric, resamples=100, seed=9) == bootstrap_metric(
        values, metric, resamples=100, seed=9
    )
