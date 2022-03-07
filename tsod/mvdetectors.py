from typing import Union

import pandas as pd
import numpy as np
import typing

from .base import Detector
from .custom_exceptions import (
    NoRangeDefinedError,
    WrongInputSizeError,
    InvalidArgumentError,
)


def make_vector_broadcastable(value, n_data_rows):
    if value is None:
        return None
    if len(value.shape) > 0:
        if len(value) != n_data_rows:
            raise WrongInputSizeError(
                "The number of rows in the input data must match the number of "
                "values specified for min and max if more than one value is given for min/max."
            )
    broadcastable_value = value
    if len(value.shape) == 1:
        broadcastable_value = value[..., np.newaxis]
    return broadcastable_value


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
    quantiles : list[2]
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

    >>> detector = MVRangeDetector(quantiles=[0.001,0.999])
    >>> detector.fit(normal_data_with_some_outliers)
    >>> anomalies = detector.detect(normal_data_with_some_outliers)"""

    def __init__(self, min_value=-np.inf, max_value=np.inf, quantiles=None):
        super().__init__()

        if min_value is not None:
            min_value = np.array(min_value)
            if len(min_value.shape) > 1:
                raise InvalidArgumentError("min_value ", " a float or 1D array_like.")

        if max_value is not None:
            max_value = np.array(max_value)
            if len(max_value.shape) > 1:
                raise InvalidArgumentError("max_value ", " a float or 1D array_like.")

        if (min_value is not None) and (max_value is not None):
            if np.array([min_value > max_value]).any():
                raise InvalidArgumentError(
                    "For all values in min_value and max_value ",
                    " the min must be less than max.",
                )

        self._min = min_value

        self._max = max_value

        if quantiles is None:
            self.quantiles = [0.0, 1.0]
        else:
            if not (0.0 <= quantiles[0] <= 1.0):
                raise InvalidArgumentError(
                    "Values in quantile_prob_cut_offs",
                    " between 0 and 1, both inclusive.",
                )
            if not (0.0 <= quantiles[1] <= 1.0):
                raise InvalidArgumentError(
                    "Values in quantile_prob_cut_offs",
                    " between 0 and 1, both inclusive.",
                )
            self.quantiles = [np.min(quantiles), np.max(quantiles)]

    def _fit(self, data):
        """Set min and max based on data.

        Parameters
        ----------
        data :  pd.DataFrame
                Time series data with time over columns.
        """
        super().validate(data)

        if isinstance(data, pd.Series):
            values_at_quantiles = np.nanquantile(data, self.quantiles)
        else:
            values_at_quantiles = np.nanquantile(data, self.quantiles, axis=1)

        self._min = values_at_quantiles[0]
        self._max = values_at_quantiles[1]

        return self

    def _detect(self, data: typing.Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Detect anomalies outside range"""

        if (self._min is None) and (self._max is None):
            raise NoRangeDefinedError(
                "Both min and max are None. At least one of them must be set."
            )

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
        return f"{super.__str__(self)}"

    def __repr__(self):
        return f"{self.__class__.__name__}(min: {self._min:.1e}, max: {self._max:.1e})"


class MVCorrelationDetector(Detector):
    def __init__(self, window_size=4, z_value=1.96):
        super().__init__()

        self.z_value = z_value
        self.window_size = window_size
        self.n_ts = None

        if self.window_size < 4:
            raise InvalidArgumentError(
                "window_size",
                " must be at least 4 to avoid infinite correlation standard deviation calculation.",
            )

        if self.z_value < 0:
            raise InvalidArgumentError("z_value", " must be positive.")

    def _fit(self, data):
        super().validate(data)

        n_obs = data.shape[1]
        self.n_ts = data.shape[0]
        corrs = np.corrcoef(data)

        upper_triangle_corrs = self.extract_upper_triangle_no_diag(corrs)
        lower, upper = self.calculate_correlation_confidence_intervals(
            upper_triangle_corrs, n_obs
        )

        self._lower = lower
        self._upper = upper
        return self

    def calculate_correlation_confidence_intervals(self, corrs, n_obs):
        # https://www.statskingdom.com/correlation-confidence-interval-calculator.html#:~:text=The%20correlation%20confidence%20interval%20is,exact%20value%20of%20the%20correlation.
        corr_transformed_space = np.arctanh(corrs)
        std_transformed_space = 1 / np.sqrt(n_obs - 3)
        lower_transformed_space = (
            corr_transformed_space - self.z_value * std_transformed_space
        )
        upper_transformed_space = (
            corr_transformed_space + self.z_value * std_transformed_space
        )
        lower = np.tanh(lower_transformed_space)
        upper = np.tanh(upper_transformed_space)
        return lower, upper

    def _detect(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        np_rolling_corr_without_ones = self.get_rolling_correlation(data)

        lower, upper = self.calculate_correlation_confidence_intervals(
            np_rolling_corr_without_ones, self.window_size
        )

        is_outside_expected_interval = (upper < self._lower) | (lower > self._upper)
        return pd.DataFrame(is_outside_expected_interval)

    def get_rolling_correlation(self, data):
        _, n_obs = data.shape
        standard_rolling_corr = data.T.rolling(self.window_size, center=False).corr()
        np_rolling_corr = np.array(standard_rolling_corr).reshape(
            (n_obs, self.n_ts, self.n_ts)
        )
        np_rolling_corr_without_ones = self.extract_upper_triangle_no_diag(
            np_rolling_corr
        )
        return np_rolling_corr_without_ones

    def extract_upper_triangle_no_diag(self, np_rolling_corr):
        upper_triangle_mask = np.zeros((self.n_ts, self.n_ts))
        triangle_indices = np.tril_indices(3, k=-1, m=3)
        upper_triangle_mask[triangle_indices] = 1
        np_rolling_corr_without_ones = np_rolling_corr[..., upper_triangle_mask == 1]
        return np_rolling_corr_without_ones
