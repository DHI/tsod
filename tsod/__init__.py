from .detectors import (
    RangeDetector,
    DiffDetector,
    ConstantGradientDetector,
    GradientDetector,
    ConstantValueDetector,
    CombinedDetector,
    RollingStandardDeviationDetector,
)

from .mvdetectors import MVRangeDetector, MVCorrelationDetector

from .base import load

__version__ = "0.1.4"
