"""Simple univariate anomaly detectors"""

from collections.abc import Sequence
import pandas as pd
import numpy as np

from .base import Detector


def _gradient(data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "GradientDetector requires a DatetimeIndex. "
            f"Got {type(data.index).__name__} instead."
        )

    dt = data.index.to_series().diff(periods).dt.total_seconds() * np.sign(periods)
    if dt.min() < 1e-15:
        raise ValueError("Index must be monotonically increasing")

    # Broadcast division with dataframe correctly
    return data.diff(periods=periods).div(dt, axis=0)


class CombinedDetector(Detector, Sequence):
    """Combine detectors.

    It is possible to combine several anomaly detection strategies into a combined detector.
    Anomalies are detected if ANY of the constituent detectors flags an anomaly (OR logic).

    Parameters
    ----------
    detectors : list of Detector
        List of detector instances to combine.

    Examples
    --------
    >>> normal_data = pd.Series(np.random.normal(size=100))
    >>> abnormal_data = pd.Series(np.random.normal(size=100))
    >>> abnormal_data[[2, 6, 15, 57, 60, 73]] = 5

    >>> anomaly_detector = CombinedDetector([RangeDetector(), DiffDetector()])
    >>> anomaly_detector.fit(normal_data)
    >>> detected_anomalies = anomaly_detector.detect(abnormal_data)
    """

    def __init__(self, detectors):
        super().__init__()

        for detector in detectors:
            if not isinstance(detector, Detector):
                raise ValueError(
                    f"""{detector} is not a Detector.
                     Did you forget to create an instance, e.g. ConstantValueDetector()?"""
                )

        self._detectors = detectors

    def _fit(self, data: pd.Series):
        for detector in self._detectors:
            detector.fit(data)
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        # NaN handling: True | NaN -> True, False | NaN -> NaN
        result = self._detectors[0].detect(data)
        for detector in self._detectors[1:]:
            result = result | detector.detect(data)
        return result

    def __getitem__(self, index):
        return self._detectors[index]

    def __len__(self):
        return len(self._detectors)


class RangeDetector(Detector):
    """Detect values outside range.

    Parameters
    ----------
    min_value : float, default=-np.inf
        Minimum value threshold.
    max_value : float, default=np.inf
        Maximum value threshold.
    quantiles : list of float, optional
        Quantiles to use for determining min and max during fit.
        Default is [0.0, 1.0], which corresponds to absolute min and max values.
        Use values like [0.001, 0.999] to exclude extreme outliers.

    Examples
    --------
    >>> normal_data = pd.Series(np.random.normal(size=100))
    >>> abnormal_data = pd.Series(np.random.normal(size=100))
    >>> abnormal_data[[2, 6, 15, 57, 60, 73]] = 5
    >>> normal_data_with_some_outliers = pd.Series(np.random.normal(size=100))
    >>> normal_data_with_some_outliers[[12, 13, 20, 90]] = 7

    >>> detector = RangeDetector(min_value=0.0, max_value=2.0)
    >>> anomalies = detector.detect(abnormal_data)

    >>> detector = RangeDetector()
    >>> detector.fit(normal_data) # min, max inferred from normal data
    >>> anomalies = detector.detect(abnormal_data)

    >>> detector = RangeDetector(quantiles=[0.001,0.999])
    >>> detector.fit(normal_data_with_some_outliers)
    >>> anomalies = detector.detect(abnormal_data)
    """

    def __init__(self, min_value=-np.inf, max_value=np.inf, quantiles=None):
        super().__init__()

        self._min = min_value

        self._max = max_value

        if quantiles is None:
            self._quantiles = [0.0, 1.0]
        else:
            assert 0.0 <= quantiles[0] <= 1.0
            assert 0.0 <= quantiles[1] <= 1.0
            self._quantiles = quantiles

    def _fit(self, data: pd.Series):
        quantiles = np.nanquantile(data, self._quantiles)
        self._min = quantiles.min()
        self._max = quantiles.max()

        assert self._max >= self._min
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies outside range"""

        if self._max is None:
            return data < self._min

        if self._min is None:
            return data > self._max

        return (data < self._min) | (data > self._max)

    def __str__(self):
        return f"{self.__class__.__name__}(min: {self._min:.1e}, max: {self._max:.1e})"


class DiffDetector(Detector):
    """Detect sudden shifts in data, irrespective of time axis.

    Parameters
    ----------
    max_diff : float, default=np.inf
        Maximum change threshold between consecutive points.
    direction : {'both', 'positive', 'negative'}, default='both'
        Direction of change to detect. 'positive' detects only increases,
        'negative' detects only decreases, 'both' detects changes in either direction.

    See Also
    --------
    GradientDetector : Similar functionality but considers actual time between data points.
    """

    def __init__(self, max_diff=np.inf, direction="both"):
        super().__init__()
        self._max_diff = max_diff

        valid_directions = ("both", "positive", "negative")
        if direction in valid_directions:
            self._direction = direction
        else:
            raise ValueError(
                f"Selected direction, '{direction}' is not a valid direction. Valid directions are: {valid_directions}"
            )

    def _fit(self, data: pd.Series):
        data_diff = data.diff()

        self._max_diff = data_diff.max()
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._direction == "both":
            return (data.diff()).abs() > self._max_diff
        elif self._direction == "positive":
            return data.diff() > self._max_diff
        else:
            return data.diff() < -self._max_diff

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self._max_diff}, direction:{self._direction})"
        )


class RollingStandardDeviationDetector(Detector):
    """Detect large variations.

    Parameters
    ----------
    window_size : int, default=10
        Number of data points to evaluate over.
    max_std : float, default=np.inf
        Maximum standard deviation to accept as normal.
    center : bool, default=True
        If True, set the labels at the center of the window.
    """

    def __init__(self, window_size=10, max_std=np.inf, center=True):
        super().__init__()
        self._window_size = window_size
        self._max_std = max_std
        self._center = center

    def _fit(self, data: pd.Series):
        self._max_std = data.rolling(self._window_size).std().max()

        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        anomalies = (
            data.rolling(self._window_size, center=self._center).std() > self._max_std
        )

        # anomalies = anomalies.astype(int).diff() > 0  # only take positive edges
        anomalies.iloc[0, :] = False  # first element cannot be determined by diff
        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}(window_size:{self._window_size}, max_std:{self._max_std})"


class ConstantValueDetector(Detector):
    """Detect contiguous periods of constant values within a configurable time window.

    Commonly caused by sensor failures, which get stuck at a constant level.

    Parameters
    ----------
    window_size : int, default=3
        Number of consecutive points to evaluate.
    threshold : float, default=1e-7
        Maximum variation (max - min) within window to consider constant.
    """

    def __init__(self, window_size: int = 3, threshold: float = 1e-7):
        super().__init__()

        # Validate input
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}")

        self._threshold = threshold
        self._window_size = window_size

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def window_size(self) -> int:
        return self._window_size

    def _fit(self, data: pd.Series):
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect constant values in single column or multiple columns."""

        # Apply detection to each column independently
        return data.apply(self._detect_single_column, axis=0)

    def _detect_single_column(self, data: pd.Series) -> pd.Series:
        """Detect constant values in a single column."""
        # Early exit for windows size larger than data
        if self.window_size >= data.shape[0]:
            return pd.Series(False, index=data.index)

        # Create shifted versions for comparison
        comparisons = [
            ((data - data.shift(i)).abs() <= self.threshold)
            for i in range(1, self.window_size)
        ]
        constant_detected = pd.concat(comparisons, axis=1).all(axis=1)

        # Use convolution-like approach to expand detections
        detections = constant_detected.astype(np.int64).to_numpy()
        kernel = np.ones(self._window_size, dtype=int)

        # Convolve to set all points in the window to anomalies
        expanded_full = np.convolve(detections, kernel, mode="full")

        # Remove boundary effects and padded data from convolution
        start_idx = self._window_size - 1
        expanded = expanded_full[start_idx : start_idx + len(detections)] > 0
        return pd.Series(expanded, index=data.index, name=data.name)

    def __str__(self):
        return f"{self.__class__.__name__}(window_size: {self.window_size}, threshold: {self.threshold})"


class ConstantGradientDetector(ConstantValueDetector):
    """Detect constant gradients.

    Typically caused by linear interpolation over a long interval.

    Parameters
    ----------
    window_size : int, default=3
        Minimum window size to consider as anomaly.
    """

    def __init__(self, window_size: int = 3):
        super().__init__(window_size=window_size)

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        gradient = _gradient(data, periods=1)
        s1 = super()._detect(gradient)
        gradient = _gradient(data, periods=-1)
        s2 = super()._detect(gradient)
        return s1 | s2

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size})"


class GradientDetector(Detector):
    """Detect abrupt changes in time series data.

    Requires data with a DatetimeIndex. Calculates rate of change per second.

    Parameters
    ----------
    max_gradient : float, default=np.inf
        Maximum rate of change per second.
    direction : {'both', 'positive', 'negative'}, default='both'
        Direction of change to detect. 'positive' detects only increases,
        'negative' detects only decreases, 'both' detects changes in either direction.
    """

    def __init__(self, max_gradient=np.inf, direction="both"):
        super().__init__()
        self._max_gradient = max_gradient
        valid_directions = ("both", "positive", "negative")
        if direction in valid_directions:
            self._direction = direction
        else:
            raise ValueError(
                f"""Selected direction, '{direction}' is not a valid direction.
                 Valid directions are: {valid_directions}"""
            )

    def _fit(self, data: pd.Series):
        # Validate that the data has a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "GradientDetector requires a DatetimeIndex. "
                f"Got {type(data.index).__name__} instead."
            )
        self._max_gradient = np.max(np.abs(_gradient(data.to_frame())))
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        gradient = _gradient(data)
        if self._direction == "negative":
            return gradient < -self._max_gradient
        elif self._direction == "positive":
            return gradient > self._max_gradient
        else:
            return gradient.abs() > self._max_gradient

    def __str__(self):
        max_grad_hr = self._max_gradient * 3600.0
        return (
            f"{self.__class__.__name__}({max_grad_hr}/hr, direction:{self._direction})"
        )
