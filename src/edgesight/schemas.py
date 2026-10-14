"""Shared, dependency-light contracts for inference and decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InferenceMode(str, Enum):
    DEMO = "demo"
    REAL = "real"


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    label: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        x1, y1, x2, y2 = self.xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError("xyxy coordinates are inverted")


@dataclass(frozen=True)
class AnomalyResult:
    score: float
    map: Any | None = None
    threshold: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def anomalous(self) -> bool | None:
        return None if self.threshold is None else self.score >= self.threshold


@dataclass(frozen=True)
class InferenceResult:
    source: str | Path | None
    mode: InferenceMode
    detections: tuple[Detection, ...] = ()
    anomaly: AnomalyResult | None = None
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityDecision:
    decision: Decision
    reason: str
    confidence: float
    inference: InferenceResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
