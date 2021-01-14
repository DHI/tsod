import numpy as np
import pandas as pd
from numba import jit

GAUSSIAN_SCALE_FACTOR = 1.4826  # k = 1/Phi^(-1)(3/4)
# Choosing 3/4 then +-MAD covers 50% of the standard normal cumulative distribution function.


def median_absolute_deviation(x):
    """ Calculate median absolute deviation (MAD) from the window's median. """
    return np.median(np.abs(x - np.median(x)))


def hampel(ts, window_size=5, n=3, k=GAUSSIAN_SCALE_FACTOR):
    """ Median absolute deviation (MAD) outlier in Time Series

    Parameters
    ----------
    n : float
        threshold, default is 3 (Pearson's rule)
    window_size : int
        total window size will be computed as 2*window_size + 1
    ts : pd.Series
    k : float
        Constant scale factor dependent on distribution. Default is normal distribution.
    """

    if type(ts) != pd.Series:
        raise ValueError("Timeseries must be of type pandas.Series.")

    if type(window_size) != int:
        raise ValueError("Window size must be of type integer.")
    else:
        if window_size <= 0:
            raise ValueError("Window size must be greater than 0.")

    if type(n) != int:
        raise ValueError("Threshold must be of type integer.")
    else:
        if n < 0:
            raise ValueError("Threshold must be positive.")

    ts_cleaned = ts.copy()

    rolling_ts = ts_cleaned.rolling(window_size * 2, center=True)
    rolling_median = rolling_ts.median().fillna(method='bfill').fillna(method='ffill')
    rolling_sigma = k * (rolling_ts.apply(median_absolute_deviation).fillna(method='bfill').fillna(method='ffill'))

    outlier_indices = list(
        np.array(np.where(np.abs(ts_cleaned - rolling_median) >= (n * rolling_sigma))).flatten())
    ts_cleaned[outlier_indices] = rolling_median[outlier_indices]

    return ts_cleaned


@jit(nopython=True)
def hampel_filter_forloop_numba(input_series, window_size, n_sigmas=3):
    """
    Hampel filter implementation that works on numpy arrays, implemented with numba. Snatched from this implementation:
    https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/outlier_detection_hampel_filter.ipynb

    Parameters
    ----------
    input_series: numpy.ndarray
    window_size: int
        The window range is from [(i - window_size):(i + window_size)], so window_size is half of the
        window, counted in number of array elements (as opposed to specify a time span, which is not
        supported by this implementation)
    n_sigmas: float
        The threshold for marking an outlier. A low threshold "narrows" the band within which values are deemed as
        outliers.

    Returns
    -------
    new_series: numpy.ndarray
        series with outliers replaced by rolling median
    indices: list
        List of indices with detected outliers
    """

    n = len(input_series)
    new_series = input_series.copy()
    k = GAUSSIAN_SCALE_FACTOR
    indices = []

    for i in range(window_size, (n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if np.abs(input_series[i] - x0) > n_sigmas * S0:
            new_series[i] = x0
            indices.append(i)

    return new_series, indices


def hampel_numba(df, k=7, t0=3):
    """
    Wraps the numba implementation of the Hampel filter to work on dataframes with datetime index.

    Parameters
    ----------
    df : pandas.DataFrame
        pandas series of values from which to remove outliers. The dataframe must have 'date' as index
        and one column named 'value'
    k : int
        size of window (including the sample; 7 is equal to 3 on either side of value)
    t0 : float
        The threshold for marking an outlier. A low threshold "narrows" the band within which values are deemed as
        outliers.

    Returns
    -------
    df_outliers : pandas.DataFrame
        Series containing only the outliers
    df_clean : pandas.DataFrame
        Series with outliers replaced by rolling median
    """
    vals = df.to_numpy().flatten()

    window_size = int(k / 2)
    clean, idx = hampel_filter_forloop_numba(vals, window_size, t0)

    df_clean = pd.DataFrame(data={'value': clean}, index=df.index)

    outlier_idx = df.index[idx]  # get outlier indices in terms of datetime
    df_outliers = df.loc[outlier_idx]

    return df_outliers, df_clean


def hampel_akf(df, k=7, t0=3):
    """
    Hampel filter implementation, which does the calculations directly on pandas.DataFrames.
    Snatched from
    https://stackoverflow.com/questions/46819260/filtering-outliers-how-to-make-median-based-hampel-function-faster
    and modified to also return dataframe with outliers.

    This implementation is much, much slower than the numba implementation.

    Parameters
    __________
    vals: pandas.DataFrame
        pandas series of values from which to remove outliers. The dataframe must have 'date' as index
        and one column named 'value'
    k: int
        size of window (including the sample; 7 is equal to 3 on either side of value)
    t0: float
        The threshold for marking an outlier. A low threshold "narrows" the band within which values are deemed as
        outliers.

    Returns
    -------
    df_outliers : pandas.DataFrame
        Series containing only the outliers
    df_clean : pandas.DataFrame
        Series with outliers replaced by rolling median
    """
    vals = df['value']

    k = GAUSSIAN_SCALE_FACTOR
    rolling_median = vals.rolling(window=k, center=True).median()
    rolling_MAD = vals.rolling(window=k, center=True).apply(median_absolute_deviation)
    threshold = t0 * k * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Comment in the StackOverflow code:
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    # Mask with outlier positions, e.g. [ 0 1 0 1 1]
    outlier_mask = difference > threshold
    outlier_idx = df.index[outlier_mask]

    # make a new dataframe which contains only the outlier rows
    df_outliers = df.filter(axis=0, items=outlier_idx)

    # make a new dataframe where the outliers have been replaced by the rolling median
    clean = vals.copy()
    clean[outlier_mask] = rolling_median[outlier_mask]
    # df_clean = pd.DataFrame(data = {'date': df['date'], 'value': clean})
    df_clean = pd.DataFrame(data={'value': clean}, index=df.index)

    return df_outliers, df_clean
