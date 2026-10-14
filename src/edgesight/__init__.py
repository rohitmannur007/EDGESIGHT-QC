"""EdgeSight-QC core library.

The package keeps heavyweight computer-vision dependencies optional.  Importing
``edgesight`` therefore works in data-validation and evaluation environments
that do not have PyTorch or Ultralytics installed.
"""

from .config import EdgeSightConfig, load_config
from .schemas import (
    AnomalyResult,
    Decision,
    Detection,
    InferenceMode,
    InferenceResult,
    QualityDecision,
)

__all__ = [
    "AnomalyResult",
    "Decision",
    "Detection",
    "EdgeSightConfig",
    "InferenceMode",
    "InferenceResult",
    "QualityDecision",
    "load_config",
]

__version__ = "0.1.0"
