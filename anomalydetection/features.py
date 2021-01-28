import pandas as pd


def lag_time_series(time_series: pd.Series, lags):
    """ Create lagged time series features.

    Parameters
    ----------
    time_series : pd.Series
    lags : list[int]
        List of lags

    Returns
    -------
    pd.DataFrame
        Lagged time series features.
    """
    lagged_time_series = {}
    for lag in lags:
        lagged_time_series[str(lag)] = time_series.shift(lag)

    return pd.concat(lagged_time_series, axis=1)
