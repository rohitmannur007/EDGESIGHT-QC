"""Joint detector/anomaly operating-point optimization."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class CostPoint:
    detector_threshold: float
    anomaly_threshold: float
    false_accepts: int
    false_rejects: int
    reviews: int
    false_alarm_rate: float
    total_cost: float
    expected_cost_per_1000: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostOptimizationResult:
    status: str
    selected: CostPoint | None
    evaluated_points: int
    feasible_points: int
    max_false_alarm_rate: float
    reason: str | None = None
    frontier: tuple[CostPoint, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected"] = self.selected.as_dict() if self.selected else None
        value["frontier"] = [point.as_dict() for point in self.frontier]
        return value


def optimize_joint_thresholds(
    detector_scores: Iterable[float],
    anomaly_scores: Iterable[float],
    labels: Iterable[int | bool],
    *,
    detector_thresholds: Iterable[float] | None = None,
    anomaly_thresholds: Iterable[float] | None = None,
    max_false_alarm_rate: float = 0.05,
    missed_defect_cost: float = 100.0,
    false_reject_cost: float = 5.0,
    review_cost: float = 1.0,
    inspection_cost: float = 0.02,
    downstream_escape_cost: float = 250.0,
) -> CostOptimizationResult:
    """Sweep joint thresholds, minimizing total cost subject to a FAR ceiling.

    A sample is rejected when either model triggers. When both trigger it is
    counted as a review, matching the inspection pipeline's routing policy.
    """
    detector = [float(value) for value in detector_scores]
    anomaly = [float(value) for value in anomaly_scores]
    truth = [bool(value) for value in labels]
    if not detector or len(detector) != len(anomaly) or len(detector) != len(truth):
        raise ValueError("scores and labels must be non-empty and have equal lengths")
    if not 0 <= max_false_alarm_rate <= 1:
        raise ValueError("max_false_alarm_rate must be between 0 and 1")
    if (
        min(
            missed_defect_cost,
            false_reject_cost,
            review_cost,
            inspection_cost,
            downstream_escape_cost,
        )
        < 0
    ):
        raise ValueError("costs must be non-negative")
    negatives = sum(not label for label in truth)
    if negatives == 0:
        return CostOptimizationResult(
            "NOT_SUPPORTED",
            None,
            0,
            0,
            max_false_alarm_rate,
            reason="false-alarm constraint requires at least one non-defective label",
        )
    detector_grid = sorted(set(float(value) for value in (detector_thresholds or detector)))
    anomaly_grid = sorted(set(float(value) for value in (anomaly_thresholds or anomaly)))
    detector_grid.append(math.nextafter(max(detector_grid), math.inf))
    anomaly_grid.append(math.nextafter(max(anomaly_grid), math.inf))
    points: list[CostPoint] = []
    feasible: list[CostPoint] = []
    for detector_threshold, anomaly_threshold in product(detector_grid, anomaly_grid):
        detector_trigger = [score >= detector_threshold for score in detector]
        anomaly_trigger = [score >= anomaly_threshold for score in anomaly]
        rejected = [
            detector_hit or anomaly_hit
            for detector_hit, anomaly_hit in zip(detector_trigger, anomaly_trigger, strict=True)
        ]
        false_accepts = sum(
            label and not reject for label, reject in zip(truth, rejected, strict=True)
        )
        false_rejects = sum(
            not label and reject for label, reject in zip(truth, rejected, strict=True)
        )
        reviews = sum(
            detector_hit and anomaly_hit
            for detector_hit, anomaly_hit in zip(detector_trigger, anomaly_trigger, strict=True)
        )
        total_cost = (
            false_accepts * (missed_defect_cost + downstream_escape_cost)
            + false_rejects * false_reject_cost
            + reviews * review_cost
            + len(truth) * inspection_cost
        )
        point = CostPoint(
            detector_threshold,
            anomaly_threshold,
            false_accepts,
            false_rejects,
            reviews,
            false_rejects / negatives,
            total_cost,
            total_cost / len(truth) * 1000,
        )
        points.append(point)
        if point.false_alarm_rate <= max_false_alarm_rate:
            feasible.append(point)
    if not feasible:
        return CostOptimizationResult(
            "NOT_SUPPORTED",
            None,
            len(points),
            0,
            max_false_alarm_rate,
            reason="no threshold pair satisfies the false-alarm constraint",
        )
    feasible.sort(
        key=lambda point: (
            point.total_cost,
            point.false_alarm_rate,
            -point.detector_threshold,
            -point.anomaly_threshold,
        )
    )
    return CostOptimizationResult(
        "OK",
        feasible[0],
        len(points),
        len(feasible),
        max_false_alarm_rate,
        frontier=tuple(feasible),
    )
