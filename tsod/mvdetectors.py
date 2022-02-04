import pandas as pd
import numpy as np
import typing

from .base import Detector
from .custom_exceptions import NoRangeDefinedError, WrongInputSize


def make_vector_broadcastable(function_input, n_data_rows):
    if function_input is not None:
        if len(function_input.shape) > 0:
            if len(function_input) != n_data_rows:
                raise WrongInputSize(
                    "The number of rows in the input data must match the number of "
                    "values specified for min and max if more than one value is given for min/max.")
    min_comparison = function_input
    if len(function_input.shape) == 1:
        min_comparison = function_input[..., np.newaxis]
    return min_comparison


class MVRangeDetector(Detector):
    """
    Detect values outside range.

    NaN values are not marked as anomalies.

    Parameters
    ----------
    min_value : float, List, np.array
        Minimum value threshold.
    max_value : float, List, np.array
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

        min_value = np.array(min_value)
        assert len(min_value.shape) <= 1

        max_value = np.array(max_value)
        assert len(max_value.shape) <= 1

        assert np.array([min_value <= max_value]).all()

        self._min = min_value

        self._max = max_value

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
        data :  pd.DataFrame
                Time series data with time over columns.
        """
        super().validate(data)

        quantiles = np.nanquantile(data, self.quantile_prob_cut_offs, axis=1)
        self._min = quantiles[0]
        self._max = quantiles[1]

        return self

    def _detect(self, data: typing.Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Detect anomalies outside range"""

        if (self._min is None) and (self._max is None):
            raise NoRangeDefinedError("Both min and max are None. At least one of them must be set.")

        if len(data.shape) == 1:
            n_data_rows = 1
        else:
            n_data_rows = data.shape[0]

        min_comparison = make_vector_broadcastable(self._min, n_data_rows)
        max_comparison = make_vector_broadcastable(self._max, n_data_rows)

        if self._max is None:
            return data < min_comparison

        if self._min is None:
            return data > max_comparison

        return (data < min_comparison) | (data > max_comparison)

    def __str__(self):

        return f"{super.__str__(self)}{self._min}, {self._max})"

    def __repr__(self):
        return f"{self.__class__.__name__}(min: {self._min:.1e}, max: {self._max:.1e})"