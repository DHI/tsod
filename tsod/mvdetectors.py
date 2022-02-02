import pandas as pd
import numpy as np
from typing import Union

from .base import Detector


class MVRangeDetector(Detector):
    """
    Detect values outside range.

    If one or more time series is out of range, is is detected as an anomaly. Note that this implies that the same range
    is used for all time series.

    Parameters
    ----------
    min_value : float
        Minimum value threshold.
    max_value : float
        Maximum value threshold.
    quantile_prob_cut_offs : list[2]
                Default quantiles [0, 1]. Same as min and max value.

    Examples
    ---------
    >>> n_obs = 100
    >>> normal_data = pd.DataFrame(np.random.normal(size=[3, n_obs]))
    >>> abnormal_data = pd.DataFrame(np.random.normal(size=[3, n_obs]))
    >>> abnormal_data.iloc[0, [2, 6, 15, 57, 60, 73]] = 5
    >>> normal_data_with_some_outliers = pd.DataFrame(np.random.normal(size=[3, n_obs]))
    >>> normal_data_with_some_outliers.iloc[0, [12, 13, 20, 90]] = 7

    >>> detector = MVRangeDetector(min_value=0.0, max_value=2.0)
    >>> anomalies = detector.detect(abnormal_data)

    >>> detector = MVRangeDetector()
    >>> detector.fit(normal_data) # min, max inferred from normal data
    >>> anomalies = detector.detect(abnormal_data)

    >>> detector = MVRangeDetector(quantile_prob_cut_offs=[0.001,0.999])
    >>> detector.fit(normal_data_with_some_outliers)
    >>> anomalies = detector.detect(normal_data_with_some_outliers)"""

    def __init__(self, min_value=-np.inf, max_value=np.inf, quantile_prob_cut_offs=None):
        super().__init__()

        self._min = min_value

        self._max = max_value

        assert self._min <= self._max

        if quantile_prob_cut_offs is None:
            self.quantile_prob_cut_offs = [0.0, 1.0]
        else:
            assert 0.0 <= quantile_prob_cut_offs[0] <= 1.0
            assert 0.0 <= quantile_prob_cut_offs[1] <= 1.0
            self.quantile_prob_cut_offs = [np.min(quantile_prob_cut_offs), np.max(quantile_prob_cut_offs)]

    def _fit(self, data):
        """Set min and max based on data.

        Parameters
        ----------
        data :  pd.Series
                Normal time series data.
        """
        super().validate(data)

        quantiles = np.quantile(data.dropna(), self.quantile_prob_cut_offs)
        self._min = quantiles[0]
        self._max = quantiles[1]

        return self

    def _detect(self, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Detect anomalies outside range"""

        minimum_values = data.min(axis=0)
        maximum_values = data.max(axis=0)

        if self._max is None:
            return minimum_values < self._min

        if self._min is None:
            return maximum_values > self._max

        return (minimum_values < self._min) | (maximum_values > self._max)

    def __str__(self):

        return f"{super.__str__(self)}{self._min}, {self._max})"

    def __repr__(self):
        return f"{self.__class__.__name__}(min: {self._min:.1e}, max: {self._max:.1e})"
