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
            "Gradient calculation requires a DatetimeIndex. "
            f"Got {type(data.index).__name__} instead."
        )

    dt = data.index.to_series().diff(periods).dt.total_seconds() * np.sign(periods)
    if dt.min() < 1e-15:
        raise ValueError("Index must be monotonically increasing")

    # Broadcast division with dataframe correctly
    return data.diff(periods=periods).div(dt, axis=0)


def _rolling_slope(
    data: pd.DataFrame, window_size: int, center: bool = False
) -> pd.DataFrame:
    """Slope of a least-squares straight line fitted in a rolling window.

    The slope of an ordinary least-squares fit can be written in terms of sums
    of t, y, t*y and t**2, so it is evaluated with rolling sums only, i.e.
    without fitting a line per window. Because the actual time stamps are used,
    non-equidistant data is handled correctly and the slope is a rate per
    second. Missing values are excluded from the fit rather than propagated, so
    a window containing NaN still yields a slope as long as it holds at least
    two valid points.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data, must have a DatetimeIndex.
    window_size : int
        Number of points in the rolling window.
    center : bool, default=False
        If True, set the labels at the center of the window instead of at its
        trailing edge.

    Returns
    -------
    pd.DataFrame
        Slope per second. Windows that are not fully populated yield NaN.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            "Slope calculation requires a DatetimeIndex. "
            f"Got {type(data.index).__name__} instead."
        )

    if data.empty:
        return data.astype(float)

    seconds = pd.Series((data.index - data.index[0]).total_seconds(), index=data.index)
    if len(seconds) > 1 and seconds.diff().iloc[1:].min() < 1e-15:
        raise ValueError("Index must be monotonically increasing")

    # Express time in units of the typical sampling interval instead of seconds.
    # This keeps the sums of squares small, which matters because the
    # denominator below is a difference of two large and nearly equal numbers.
    scale = seconds.diff().iloc[1:].median() if len(seconds) > 1 else 1.0
    if not scale > 0:
        scale = 1.0
    time = seconds / scale

    def _slope(values: pd.Series) -> pd.Series:
        valid = values.notna().astype(np.float64)
        filled = values.fillna(0.0)

        def rolling_sum(series: pd.Series) -> pd.Series:
            return series.rolling(window_size, center=center).sum()

        n = rolling_sum(valid)
        sum_t = rolling_sum(time * valid)
        sum_tt = rolling_sum(time * time * valid)
        sum_y = rolling_sum(filled)
        sum_ty = rolling_sum(time * filled)

        denominator = n * sum_tt - sum_t * sum_t
        # A slope is undefined for fewer than two points, and the denominator
        # vanishes when all valid points in the window share one time stamp.
        denominator = denominator.where((n >= 2) & (denominator > 0))

        return (n * sum_ty - sum_t * sum_y) / (denominator * scale)

    return data.apply(_slope, axis=0)


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

    def __init__(self, detectors: list[Detector]):
        super().__init__()

        for detector in detectors:
            if not isinstance(detector, Detector):
                raise ValueError(
                    f"""{detector} is not a Detector.
                     Did you forget to create an instance, e.g. ConstantValueDetector()?"""
                )

        self._detectors: list[Detector] = detectors

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

    def __init__(
        self,
        min_value: float = -np.inf,
        max_value: float = np.inf,
        quantiles: list[float] | None = None,
    ):
        super().__init__()

        self._min: float = min_value
        self._max: float = max_value
        self._quantiles: list[float]

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

    def __init__(self, max_diff: float = np.inf, direction: str = "both"):
        super().__init__()
        self._max_diff: float = max_diff

        valid_directions = ("both", "positive", "negative")
        if direction in valid_directions:
            self._direction = direction
        else:
            raise ValueError(
                f"Selected direction, '{direction}' is not a valid direction. Valid directions are: {valid_directions}"
            )

    def _fit(self, data):
        data_diffs = data.diff()

        if self._direction == "positive":
            filtered_diffs = data_diffs[data_diffs >= 0]
        elif self._direction == "negative":
            filtered_diffs = data_diffs[data_diffs <= 0].abs()
        else:  # both
            filtered_diffs = data_diffs.abs()

        self._max_diff = filtered_diffs.max() if not filtered_diffs.empty else 0
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

    def __init__(
        self, window_size: int = 10, max_std: float = np.inf, center: bool = True
    ):
        super().__init__()
        self._window_size: int = window_size
        self._max_std: float = max_std
        self._center: bool = center

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

    def __init__(self, max_gradient: float = np.inf, direction: str = "both"):
        super().__init__()
        self._max_gradient: float = max_gradient

        valid_directions = ("both", "positive", "negative")
        if direction in valid_directions:
            self._direction = direction
        else:
            raise ValueError(
                f"""Selected direction, '{direction}' is not a valid direction.
                 Valid directions are: {valid_directions}"""
            )

    def _fit(self, data: pd.Series):
        """Set max gradient based on data."""
        gradients = _gradient(data.to_frame())

        # Filter based on direction
        if self._direction == "positive":
            filtered_gradients = gradients[gradients >= 0]
        elif self._direction == "negative":
            filtered_gradients = gradients[gradients <= 0].abs()
        else:  # both directions
            filtered_gradients = gradients.abs()

        self._max_gradient = (
            filtered_gradients.max().iloc[0] if not filtered_gradients.empty else 0
        )
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


class DriftDetector(Detector):
    """Detect slowly drifting sensors.

    Drift is a small, persistent, one-directional deviation, e.g. caused by
    biofouling or a gradual loss of calibration. Each individual step is far too
    small for `DiffDetector` or `GradientDetector` to react to, and the signal
    only leaves the interval of `RangeDetector` once the drift has become severe.
    This detector instead fits a least-squares straight line in a rolling window
    and flags the windows whose slope is too steep, which makes it sensitive to a
    trend that persists over many points.

    Requires data with a DatetimeIndex. The drift rate is a rate per second, in
    line with `GradientDetector`. Missing values are excluded from the fit rather
    than propagated, so a window containing NaN still yields a drift rate as long
    as it holds at least two valid points.

    Note that a drifting sensor and a genuine slow change in the environment look
    the same in a single time series. Consider applying the detector to a quantity
    that is expected to be stationary, such as a daily minimum, or to the
    difference between the sensor and a redundant reference.

    Parameters
    ----------
    window_size : int, default=100
        Number of data points to fit the trend over. The window should be long
        enough that noise and periodic variation average out, but short enough
        that the drift is approximately linear within it. See the notes below on
        choosing it for a periodic signal.
    max_drift_rate : float, default=np.inf
        Maximum trend to accept as normal, in units per second.
    direction : {'both', 'positive', 'negative'}, default='both'
        Direction of drift to detect. 'positive' detects only upward drift,
        'negative' only downward drift, 'both' detects drift either way.
    center : bool, default=False
        If True, set the labels at the center of the window. The default labels
        each window at its trailing edge, which is what real-time detection on a
        growing series needs.

    See Also
    --------
    GradientDetector : Detects abrupt change, i.e. a steep rate between
        neighbouring points, rather than a trend sustained over a window.

    Notes
    -----
    On a periodic signal, such as a tidal water level, the window must span
    several whole cycles. A shorter window sits on the rising or falling limb of
    the cycle, so its trend reflects the tide rather than the drift, and a
    `max_drift_rate` fitted on such windows is far too large to react to a real
    drift. As an example, for a semi-diurnal tide of 1 m amplitude a window of
    half a cycle yields a fitted rate of about 10 m/day, whereas ten cycles
    (roughly five days) brings it down to about 0.04 m/day and so makes a drift
    of a few cm/day detectable.

    Examples
    --------
    >>> time = pd.date_range("2020", periods=200, freq="1h")
    >>> normal_data = pd.Series(np.random.normal(size=200), index=time)
    >>> drifting_data = normal_data + np.linspace(0.0, 10.0, 200)

    >>> detector = DriftDetector(window_size=50)
    >>> detector.fit(normal_data)  # max drift rate inferred from normal data
    >>> anomalies = detector.detect(drifting_data)

    >>> # 1 cm per day is too much for this sensor
    >>> detector = DriftDetector(window_size=50, max_drift_rate=0.01 / 86400)
    >>> anomalies = detector.detect(drifting_data)
    """

    def __init__(
        self,
        window_size: int = 100,
        max_drift_rate: float = np.inf,
        direction: str = "both",
        center: bool = False,
    ):
        super().__init__()

        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}")
        if max_drift_rate < 0:
            raise ValueError(
                f"max_drift_rate must be non-negative, got {max_drift_rate}"
            )

        self._window_size: int = window_size
        self._max_drift_rate: float = max_drift_rate
        self._center: bool = center

        valid_directions = ("both", "positive", "negative")
        if direction in valid_directions:
            self._direction = direction
        else:
            raise ValueError(
                f"Selected direction, '{direction}' is not a valid direction. Valid directions are: {valid_directions}"
            )

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def max_drift_rate(self) -> float:
        return self._max_drift_rate

    def _slope(self, data: pd.DataFrame) -> pd.DataFrame:
        return _rolling_slope(data, self._window_size, center=self._center)

    def _fit(self, data: pd.Series):
        """Set the maximum drift rate to the steepest trend in normal data."""
        slopes = self._slope(data.to_frame()).iloc[:, 0]

        if self._direction == "positive":
            filtered_slopes = slopes[slopes >= 0]
        elif self._direction == "negative":
            filtered_slopes = slopes[slopes <= 0].abs()
        else:  # both
            filtered_slopes = slopes.abs()

        max_slope = filtered_slopes.max()
        self._max_drift_rate = 0.0 if pd.isna(max_slope) else max_slope
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        slope = self._slope(data)

        if self._direction == "positive":
            return slope > self._max_drift_rate
        elif self._direction == "negative":
            return slope < -self._max_drift_rate
        else:
            return slope.abs() > self._max_drift_rate

    def __str__(self):
        rate_per_day = self._max_drift_rate * 86400.0
        return (
            f"{self.__class__.__name__}(window_size:{self._window_size}, "
            f"max_drift_rate:{rate_per_day}/day, direction:{self._direction})"
        )
