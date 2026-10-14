"""Conservative quality-decision routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from edgesight.schemas import Decision, InferenceResult, QualityDecision


class InspectionDecision(str, Enum):
    PASS = "PASS"
    KNOWN_DEFECT = "KNOWN_DEFECT"
    UNKNOWN_DEFECT = "UNKNOWN_DEFECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass(frozen=True)
class RoutingResult:
    decision: InspectionDecision
    reason: str
    confidence: float
    review_required: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class QualityRouter:
    detection_acceptance: float = 0.70
    detection_rejection: float = 0.20
    anomaly_threshold: float = 0.50
    fail_closed: bool = True
    human_review_enabled: bool = True

    def decide(self, result: InferenceResult) -> RoutingResult:
        """Return the canonical manufacturing decision.

        Model availability is taken from engine metadata so an empty detector
        result is distinguishable from a detector that was never deployed.
        """

        configured = result.metadata.get("configured_models", {})
        detector_configured = bool(configured.get("detector", bool(result.detections)))
        anomaly_configured = bool(configured.get("anomaly", result.anomaly is not None))
        failures = dict(result.metadata.get("model_failures", {}))
        if failures:
            return self._canonical(
                InspectionDecision.HUMAN_REVIEW,
                "model execution failed",
                0.0,
                failures=failures,
            )
        if not detector_configured and not anomaly_configured:
            if self.fail_closed:
                return self._canonical(
                    InspectionDecision.HUMAN_REVIEW,
                    "no model is configured",
                    0.0,
                )
            return self._canonical(InspectionDecision.PASS, "no model is configured", 0.0)

        highest = max((item.confidence for item in result.detections), default=0.0)
        anomaly_score = result.anomaly.score if result.anomaly is not None else 0.0
        detector_trigger = detector_configured and highest >= self.detection_acceptance
        anomaly_trigger = (
            anomaly_configured
            and result.anomaly is not None
            and anomaly_score >= self.anomaly_threshold
        )
        detector_uncertain = (
            detector_configured
            and bool(result.detections)
            and self.detection_rejection < highest < self.detection_acceptance
        )
        if detector_trigger and anomaly_trigger:
            if self.human_review_enabled:
                return self._canonical(
                    InspectionDecision.HUMAN_REVIEW,
                    "detector and anomaly model both triggered",
                    max(highest, anomaly_score),
                )
            return self._canonical(
                InspectionDecision.KNOWN_DEFECT,
                "detector and anomaly model both triggered",
                highest,
            )
        if detector_trigger:
            return self._canonical(
                InspectionDecision.KNOWN_DEFECT,
                "known-defect confidence exceeded threshold",
                highest,
            )
        if anomaly_trigger:
            return self._canonical(
                InspectionDecision.UNKNOWN_DEFECT,
                "anomaly score exceeded threshold",
                anomaly_score,
            )
        if detector_uncertain and self.human_review_enabled:
            return self._canonical(
                InspectionDecision.HUMAN_REVIEW,
                "detector confidence is within the review band",
                highest,
            )
        normal_confidence = 1.0 - max(highest, anomaly_score)
        return self._canonical(
            InspectionDecision.PASS,
            "all configured checks passed",
            normal_confidence,
        )

    def _canonical(
        self,
        decision: InspectionDecision,
        reason: str,
        confidence: float,
        **metadata: Any,
    ) -> RoutingResult:
        return RoutingResult(
            decision=decision,
            reason=reason,
            confidence=max(0.0, min(1.0, confidence)),
            review_required=decision is InspectionDecision.HUMAN_REVIEW,
            metadata=metadata,
        )

    def route(self, result: InferenceResult) -> QualityDecision:
        """Compatibility adapter for the original public decision schema."""

        canonical = self.decide(result)
        legacy = {
            InspectionDecision.PASS: Decision.ACCEPT,
            InspectionDecision.KNOWN_DEFECT: Decision.REJECT,
            InspectionDecision.UNKNOWN_DEFECT: Decision.REJECT,
            InspectionDecision.HUMAN_REVIEW: Decision.REVIEW,
        }[canonical.decision]
        return QualityDecision(
            legacy,
            canonical.reason,
            canonical.confidence,
            result,
            {"canonical_decision": canonical.decision.value, **canonical.metadata},
        )

    def route_legacy(self, result: InferenceResult) -> QualityDecision:
        """Previous routing policy retained for callers needing its exact order."""

        anomaly = result.anomaly
        if anomaly is not None and anomaly.score >= self.anomaly_threshold:
            return QualityDecision(
                Decision.REJECT,
                "anomaly score exceeded threshold",
                min(1.0, anomaly.score),
                result,
            )
        if result.detections:
            highest = max(item.confidence for item in result.detections)
            if highest >= self.detection_acceptance:
                return QualityDecision(
                    Decision.REJECT,
                    "defect detector exceeded rejection confidence",
                    highest,
                    result,
                )
            if highest > self.detection_rejection:
                return QualityDecision(
                    Decision.REVIEW,
                    "detector confidence is inconclusive",
                    highest,
                    result,
                )
        if anomaly is None and not result.detections and self.fail_closed:
            return QualityDecision(
                Decision.REVIEW,
                "no usable model output",
                0.0,
                result,
            )
        normal_confidence = 1.0 - (anomaly.score if anomaly is not None else 0.0)
        return QualityDecision(
            Decision.ACCEPT,
            "all configured checks passed",
            max(0.0, min(1.0, normal_confidence)),
            result,
        )
