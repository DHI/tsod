from importlib.metadata import version
from .detectors import (
    RangeDetector,
    DiffDetector,
    ConstantGradientDetector,
    GradientDetector,
    ConstantValueDetector,
    CombinedDetector,
    RollingStandardDeviationDetector,
)

from .hampel import HampelDetector
from .base import load

try:
    __version__ = version("tsod")
except Exception:
    __version__ = "unknown"

__all__ = [
    "RangeDetector",
    "DiffDetector",
    "ConstantGradientDetector",
    "GradientDetector",
    "ConstantValueDetector",
    "CombinedDetector",
    "RollingStandardDeviationDetector",
    "load",
    "HampelDetector",
]
