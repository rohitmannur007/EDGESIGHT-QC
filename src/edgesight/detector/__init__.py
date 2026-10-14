"""Object detector adapters."""

from .yolo import MissingModelError, YoloDetector, train_yolo

__all__ = ["MissingModelError", "YoloDetector", "train_yolo"]
