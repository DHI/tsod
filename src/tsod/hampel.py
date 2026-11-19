"""Hampel detector"""

import numpy as np
import pandas as pd
from tsod.custom_exceptions import NotIntegerError, InvalidArgumentError
from tsod.detectors import Detector

# numba is a default dependency, but it is only needed by HampelDetector. Without numba,
# a stand-in jit makes the decorated function raise ImportError when it is called, so the
# rest of tsod (and creating a HampelDetector) works; only detection needs numba.
try:
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):
        def decorator(func):
            def numba_missing(*a, **k):
                raise ImportError(
                    "HampelDetector requires numba, which is not installed. "
                    "Install it with: pip install numba"
                )

            return numba_missing

        return decorator


# GAUSSIAN_SCALE_FACTOR = k = 1/Phi^(-1)(3/4)
# Choosing 3/4 as argument makes +-MAD cover 50% of the standard normal cumulative distribution function.

GAUSSIAN_SCALE_FACTOR = 1.4826


def _validate_arguments(window_size, threshold, min_mad=0.0):
    if not isinstance(window_size, (int, np.integer)) or isinstance(window_size, bool):
        raise NotIntegerError("window_size")
    else:
        if window_size <= 0:
            raise InvalidArgumentError("window_size", "nonnegative")

    if threshold < 0:
        raise InvalidArgumentError("threshold", "positive")

    if min_mad < 0:
        raise InvalidArgumentError("min_mad", "nonnegative")


@jit(nopython=True)
def _detect(
    time_series, window_size, threshold=3, k=GAUSSIAN_SCALE_FACTOR, min_mad=0.0
):
    """
    Hampel filter implementation that works on numpy arrays, implemented with numba.

    Parameters
    ----------
    time_series: numpy.ndarray
    window_size: int
        The window range is from [(i - window_size):(i + window_size)], so window_size is half of the
        window, counted in number of array elements (as opposed to specify a time span, which is not
        supported by this implementation)
    threshold: float
        The threshold for marking an outlier. A low threshold "narrows" the band within which values are deemed as
        outliers. n_sigmas
    k : float
        Constant scale factor dependent on distribution. Default is normal distribution.
    min_mad : float
        Lower limit for the scaled MAD, in data units.
    """

    # time_series_clean = time_series.copy()
    # outlier_indices = []
    is_outlier = [False] * len(time_series)

    for t in range(window_size, (len(time_series) - window_size)):
        time_series_window = time_series[(t - window_size) : (t + window_size)]
        median_in_window = np.nanmedian(time_series_window)
        mad_in_window = k * np.nanmedian(np.abs(time_series_window - median_in_window))
        mad_in_window = max(mad_in_window, min_mad)
        absolute_deviation_from_median = np.abs(time_series[t] - median_in_window)
        is_outlier[t] = absolute_deviation_from_median > threshold * mad_in_window
        # if is_outlier[t]:
        #    outlier_indices.append(t)
        #    time_series_clean[t] = median_in_window

    return is_outlier


class HampelDetector(Detector):
    """
    Hampel filter implementation that works on numpy arrays, implemented with numba.

    Parameters
    ----------
    window_size: int
        The window range is from [(i - window_size):(i + window_size)], so window_size is half of the
        window, counted in number of array elements (as opposed to specify a time span, which is not
        supported by this implementation)
    threshold: float
        The threshold for marking an outlier. A low threshold "narrows" the band within which values are deemed as
        outliers. n_sigmas, default=3.0
    min_mad: float
        Lower limit for the scaled MAD, in data units. On flat or quantised data (e.g. rain
        gauges, on/off pumps) the MAD can be 0, and then any deviation from the median is
        flagged. Setting `min_mad` to about the sensor resolution prevents this. default=0.0
    """

    def __init__(self, window_size=5, threshold=3, min_mad=0.0):
        super().__init__()
        _validate_arguments(window_size, threshold, min_mad)
        self._threshold = threshold
        self._window_size = int(window_size)
        self._min_mad = float(min_mad)

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply column-wise detection
        return data.apply(
            lambda col: _detect(
                col.to_numpy(dtype=float),
                self._window_size,
                self._threshold,
                GAUSSIAN_SCALE_FACTOR,
                getattr(self, "_min_mad", 0.0),
            ),
            axis=0,
        ).astype(bool)

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"
