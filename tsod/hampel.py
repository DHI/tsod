import numpy as np
from numba import jit

from tsod.custom_exceptions import NotInteger, InvalidArgument
from tsod.detectors import Detector

"""
GAUSSIAN_SCALE_FACTOR = k = 1/Phi^(-1)(3/4)
Choosing 3/4 as argument makes +-MAD cover 50% of the standard normal cumulative distribution function.
"""
GAUSSIAN_SCALE_FACTOR = 1.4826


def median_absolute_deviation(x):
    """ Calculate median absolute deviation (MAD) from the window's median. """
    return np.median(np.abs(x - np.median(x)))


def filter(time_series, window_size=5, threshold=3, k=GAUSSIAN_SCALE_FACTOR):
    """Detect and filter out outliers using the Hampel filter.
        Based on https://github.com/MichaelisTrofficus/hampel_filter

    Parameters
    ----------
    threshold : float
        threshold, default is 3 (Pearson's rule)
    window_size : int
        total window size will be computed as 2*window_size + 1
    time_series : pd.Series
    k : float
        Constant scale factor dependent on distribution. Default is normal distribution.
    """

    validate_arguments(window_size, threshold)

    time_series_clean = time_series.copy()
    is_outlier, outlier_indices, rolling_median = detect(
        time_series_clean, window_size, threshold, k
    )
    time_series_clean[list(outlier_indices)] = rolling_median[list(outlier_indices)]

    return is_outlier, outlier_indices, time_series_clean


def detect(time_series, window_size, threshold, k=GAUSSIAN_SCALE_FACTOR):
    """ Detect outliers using the Hampel filter. """
    rolling_ts = time_series.rolling(window_size * 2, center=True)
    rolling_median = rolling_ts.median().fillna(method="bfill").fillna(method="ffill")
    rolling_sigma = k * (
        rolling_ts.apply(median_absolute_deviation)
        .fillna(method="bfill")
        .fillna(method="ffill")
    )
    is_outlier = np.abs(time_series - rolling_median) >= (threshold * rolling_sigma)
    outlier_indices = np.array(np.where(is_outlier)).flatten()
    return is_outlier, outlier_indices, rolling_median


def validate_arguments(window_size, threshold):
    if not isinstance(window_size, int):
        raise NotInteger("window_size")
    else:
        if window_size <= 0:
            raise InvalidArgument("window_size", "nonnegative")

    if not isinstance(threshold, int):
        raise NotInteger("threshold")
    else:
        if threshold < 0:
            raise InvalidArgument("threshold", "positive")


@jit(nopython=True)
def detect_using_numba(time_series, window_size, threshold=3, k=GAUSSIAN_SCALE_FACTOR):
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
    """

    time_series_clean = time_series.copy()
    outlier_indices = []
    is_outlier = [False] * len(time_series)

    for t in range(window_size, (len(time_series) - window_size)):
        time_series_window = time_series[(t - window_size) : (t + window_size)]
        median_in_window = np.nanmedian(time_series_window)
        mad_in_window = k * np.nanmedian(np.abs(time_series_window - median_in_window))
        absolute_deviation_from_median = np.abs(time_series[t] - median_in_window)
        is_outlier[t] = absolute_deviation_from_median > threshold * mad_in_window
        if is_outlier[t]:
            outlier_indices.append(t)
            time_series_clean[t] = median_in_window

    return is_outlier, outlier_indices, time_series_clean


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
        outliers. n_sigmas
    use_numba: bool
        option to use numba for higher performance, default false
    """

    def __init__(self, window_size=5, threshold=3, use_numba=False):
        super().__init__()
        validate_arguments(window_size, threshold)
        self._threshold = threshold
        self._window_size = window_size
        self._use_numba = use_numba

    def _detect(self, data):

        if self._use_numba:
            anomalies, indices, _ = detect_using_numba(
                data.values, self._window_size, self._threshold
            )
        else:
            anomalies, indices, _ = detect(data, self._window_size, self._threshold)

        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"