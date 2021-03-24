import pandas as pd
import numpy as np


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


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
