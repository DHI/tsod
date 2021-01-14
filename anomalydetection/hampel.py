import numpy as np

'''
GAUSSIAN_SCALE_FACTOR = k = 1/Phi^(-1)(3/4)
Choosing 3/4 as argument makes +-MAD cover 50% of the standard normal cumulative distribution function.
'''
GAUSSIAN_SCALE_FACTOR = 1.4826


def median_absolute_deviation(x):
    """ Calculate median absolute deviation (MAD) from the window's median. """
    return np.median(np.abs(x - np.median(x)))


def filter(time_series, window_size=5, threshold=3, k=GAUSSIAN_SCALE_FACTOR):
    """ Detect and filter out outliers using the Hampel filter.

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
    is_outlier, outlier_indices, rolling_median = detect(time_series_clean, window_size, threshold, k)
    time_series_clean[list(outlier_indices)] = rolling_median[list(outlier_indices)]

    return is_outlier, outlier_indices, time_series_clean


def detect(time_series, window_size, threshold, k=GAUSSIAN_SCALE_FACTOR):
    """ Detect outliers using the Hampel filter. """
    rolling_ts = time_series.rolling(window_size * 2, center=True)
    rolling_median = rolling_ts.median().fillna(method='bfill').fillna(method='ffill')
    rolling_sigma = k * (rolling_ts.apply(median_absolute_deviation).fillna(method='bfill').fillna(method='ffill'))
    is_outlier = np.abs(time_series - rolling_median) >= (threshold * rolling_sigma)
    outlier_indices = np.array(np.where(is_outlier)).flatten()
    return is_outlier, outlier_indices, rolling_median


def validate_arguments(window_size, threshold):
    if type(window_size) != int:
        raise ValueError("Window size must be an integer.")
    else:
        if window_size <= 0:
            raise ValueError("Window size must be nonnegative.")

    if type(threshold) != int:
        raise ValueError("Threshold must be an integer.")
    else:
        if threshold < 0:
            raise ValueError("Threshold must be positive.")
