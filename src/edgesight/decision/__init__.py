"""Decision routing and score calibration."""

from .router import InspectionDecision, QualityRouter, RoutingResult

__all__ = [
    "InspectionDecision",
    "QualityRouter",
    "RoutingResult",
    "TemperatureScaler",
    "expected_calibration_error",
    "optimal_threshold",
]


def __getattr__(name: str):
    if name in {
        "TemperatureScaler",
        "expected_calibration_error",
        "optimal_threshold",
    }:
        from . import calibration

        return getattr(calibration, name)
    raise AttributeError(name)
