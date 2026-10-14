from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edgesight.decision import QualityRouter
from edgesight.schemas import (
    AnomalyResult,
    Decision,
    Detection,
    InferenceMode,
    InferenceResult,
)


def _result(*, detection: float | None = None, anomaly: float | None = None):
    detections = (Detection(0, detection, (0, 0, 10, 10)),) if detection is not None else ()
    anomaly_result = AnomalyResult(anomaly) if anomaly is not None else None
    return InferenceResult(None, InferenceMode.REAL, detections, anomaly_result)


def test_router_rejects_anomaly_before_detector() -> None:
    decision = QualityRouter().route(_result(detection=0.1, anomaly=0.9))
    assert decision.decision is Decision.REJECT
    assert "anomaly" in decision.reason


def test_router_triages_uncertain_and_missing_outputs() -> None:
    router = QualityRouter()
    assert router.route(_result(detection=0.4)).decision is Decision.REVIEW
    assert router.route(_result()).decision is Decision.REVIEW
    assert QualityRouter(fail_closed=False).route(_result()).decision is Decision.ACCEPT
