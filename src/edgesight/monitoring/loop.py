"""Windowed quality monitoring with durable consecutive-alert semantics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .drift import compare_distributions


def image_observations(payload: bytes) -> dict[str, float]:
    """Measure inexpensive image-quality signals from encoded image bytes."""
    import io

    import numpy as np
    from PIL import Image, ImageFilter

    with Image.open(io.BytesIO(payload)) as source:
        gray = np.asarray(source.convert("L"), dtype=np.float32)
        width, height = source.size
    edges = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    )
    pixels = max(1, width * height)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur": float(edges.var()),
        # Encoded bytes/pixel is a stable compression proxy, not a JPEG quality estimate.
        "compression": float(len(payload) / pixels),
    }


@dataclass(frozen=True)
class MonitoringOutcome:
    window_id: str | None
    sample_count: int
    reports: dict[str, dict[str, Any]]
    consecutive_drift: int
    alerted: bool
    event_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MonitoringLoop:
    """Accumulate observations, compare full windows, and alert consecutively."""

    def __init__(
        self,
        baseline: Mapping[str, Sequence[float] | Mapping[str, float]],
        *,
        store: Any | None = None,
        baseline_name: str = "default",
        baseline_version: str = "v1",
        window_size: int = 50,
        consecutive_windows: int = 2,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.2,
    ) -> None:
        if window_size < 2 or consecutive_windows < 1:
            raise ValueError("window_size must be >= 2 and consecutive_windows >= 1")
        class_reference = baseline.get("class_distribution")
        self.class_baseline = (
            {str(key): float(value) for key, value in class_reference.items()}
            if isinstance(class_reference, Mapping)
            else {}
        )
        self.baseline = {
            name: [float(value) for value in values]
            for name, values in baseline.items()
            if name != "class_distribution" and not isinstance(values, Mapping)
        }
        if (not self.baseline and not self.class_baseline) or any(
            not values for values in self.baseline.values()
        ):
            raise ValueError("baseline features must be non-empty")
        self.store = store
        self.baseline_name = baseline_name
        self.baseline_version = baseline_version
        self.window_size = window_size
        self.consecutive_windows = consecutive_windows
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self._window: dict[str, list[float]] = defaultdict(list)
        self._classes: Counter[str] = Counter()
        self._samples = 0
        self._consecutive = 0
        if store is not None:
            store.save_baseline(
                baseline_name,
                {**self.baseline, "class_distribution": self.class_baseline},
                version=baseline_version,
            )

    def observe(
        self,
        values: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MonitoringOutcome | None:
        """Add one sample; returns an outcome only when a window closes."""
        for feature in self.baseline:
            if feature in values and values[feature] is not None:
                value = float(values[feature])
                if math.isfinite(value):
                    self._window[feature].append(value)
                    if self.store is not None:
                        self.store.record_drift(feature, value, metadata)
        defect_class = values.get("class") or values.get("defect_class")
        if defect_class is not None:
            self._classes[str(defect_class)] += 1
        self._samples += 1
        if self._samples < self.window_size:
            return None
        return self.flush(metadata=metadata)

    def flush(self, *, metadata: Mapping[str, Any] | None = None) -> MonitoringOutcome | None:
        if not self._samples:
            return None
        reports: dict[str, dict[str, Any]] = {}
        for feature, baseline_values in self.baseline.items():
            current = self._window.get(feature, [])
            if current:
                reports[feature] = compare_distributions(
                    feature,
                    baseline_values,
                    current,
                    psi_threshold=self.psi_threshold,
                    ks_threshold=self.ks_threshold,
                ).as_dict()
        if self._classes:
            total = sum(self._classes.values())
            labels = set(self.class_baseline) | set(self._classes)
            observed = {label: self._classes[label] / total for label in labels}
            reference_total = sum(self.class_baseline.values())
            reference = {
                label: (
                    self.class_baseline.get(label, 0.0) / reference_total
                    if reference_total
                    else 0.0
                )
                for label in labels
            }
            total_variation = (
                sum(abs(observed[label] - reference[label]) for label in labels) / 2
                if self.class_baseline
                else None
            )
            reports["class_distribution"] = {
                "counts": dict(self._classes),
                "sample_count": total,
                "observed": observed,
                "baseline": reference,
                "total_variation": total_variation,
                "drifted": (
                    total_variation >= self.ks_threshold if total_variation is not None else False
                ),
                "note": (
                    None
                    if self.class_baseline
                    else "categorical counts persisted; a reference distribution is required"
                ),
            }
        drifted = any(bool(report["drifted"]) for report in reports.values())
        self._consecutive = self._consecutive + 1 if drifted else 0
        alerted = drifted and self._consecutive >= self.consecutive_windows
        event_ids: list[str] = []
        window_id = None
        if self.store is not None:
            window_id = self.store.record_monitoring_window(
                self.baseline_name,
                reports,
                sample_count=self._samples,
                consecutive_drift=self._consecutive,
                alerted=alerted,
                metadata=metadata,
            )
            if alerted:
                for feature, report in reports.items():
                    if report.get("drifted"):
                        event_ids.append(
                            self.store.record_drift_event(
                                feature,
                                report,
                                baseline_version=self.baseline_version,
                            )
                        )
        outcome = MonitoringOutcome(
            window_id,
            self._samples,
            reports,
            self._consecutive,
            alerted,
            tuple(event_ids),
        )
        self._window.clear()
        self._classes.clear()
        self._samples = 0
        return outcome
