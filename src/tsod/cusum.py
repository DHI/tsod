"""CUSUM drift detector"""

import numpy as np
from numba import jit
import pandas as pd

from tsod.custom_exceptions import InvalidArgumentError, NotFittedError
from tsod.detectors import Detector

# GAUSSIAN_SCALE_FACTOR = k = 1/Phi^(-1)(3/4), see tsod.hampel
GAUSSIAN_SCALE_FACTOR = 1.4826


@jit(nopython=True)
def _cusum(standardized, slack, threshold, detect_positive, detect_negative):
    """Two-sided cumulative sum control chart, implemented with numba.

    The cumulative sums accumulate how far the signal has been away from its
    expected level, discarding the part of each deviation that is smaller than
    `slack`. A deviation that is too small to be visible in any single point
    therefore still adds up until it crosses `threshold`.

    Parameters
    ----------
    standardized : numpy.ndarray
        Signal expressed in standard deviations away from its expected level.
    slack : float
        Deviation to tolerate per point, in standard deviations. Deviations
        smaller than this do not accumulate.
    threshold : float
        Decision interval. An anomaly is flagged once a cumulative sum exceeds
        this value.
    detect_positive, detect_negative : bool
        Whether upward and downward drift should be flagged.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where a cumulative sum exceeds the threshold.
    """

    n = len(standardized)
    is_anomaly = np.zeros(n, dtype=np.bool_)
    sum_positive = 0.0
    sum_negative = 0.0

    for t in range(n):
        value = standardized[t]
        if np.isnan(value):
            # A gap carries no evidence either way, so the sums are held.
            continue

        sum_positive = max(0.0, sum_positive + value - slack)
        sum_negative = min(0.0, sum_negative + value + slack)

        exceeded_positive = detect_positive and sum_positive > threshold
        exceeded_negative = detect_negative and -sum_negative > threshold
        is_anomaly[t] = exceeded_positive or exceeded_negative

    return is_anomaly


class CusumDriftDetector(Detector):
    """Detect slowly drifting sensors with a cumulative sum control chart.

    Where `DriftDetector` looks for a trend inside a window of fixed length, this
    detector accumulates evidence without forgetting it at a window edge, and so
    reacts to a persistent deviation earlier. That makes it a good fit for
    detecting a small offset or a slow drift on a signal whose normal level is
    known, and it evaluates a series in a single pass.

    The expected level and the scale of the normal data are set by `fit`, using
    the median and the median absolute deviation so that outliers in the training
    data do not inflate the scale. Unlike the textbook control chart, the
    cumulative sums are not reset when an anomaly is flagged, so that the whole
    drifted period is flagged rather than only the point at which the drift became
    detectable.

    Note that a drifting sensor and a genuine slow change in the environment look
    the same in a single time series, see the note in `DriftDetector`.

    Parameters
    ----------
    slack : float, default=0.5
        Deviation to tolerate per point, in standard deviations of the normal
        data. Conventionally set to half of the smallest shift worth detecting.
    threshold : float, default=5.0
        Decision interval, in standard deviations of the normal data. Lower values
        detect drift sooner at the cost of more false alarms.
    direction : {'both', 'positive', 'negative'}, default='both'
        Direction of drift to detect. 'positive' detects only upward drift,
        'negative' only downward drift, 'both' detects drift either way.

    See Also
    --------
    DriftDetector : Thresholds the trend within a rolling window, which gives a
        drift rate in physical units per day rather than accumulated evidence.

    Notes
    -----
    `fit` sets the expected level and scale, but not `threshold`. A control chart
    has a false alarm rate by design: with the default settings a purely noisy
    signal produces an isolated flag roughly every few hundred points, so a few
    scattered flags are expected even on data that behaves like the training data.
    Raise `threshold` to trade sensitivity for fewer false alarms, and check the
    rate on known-good data before deploying.

    Examples
    --------
    >>> time = pd.date_range("2020", periods=200, freq="1h")
    >>> normal_data = pd.Series(np.random.normal(size=200), index=time)
    >>> drifting_data = normal_data + np.linspace(0.0, 10.0, 200)

    >>> detector = CusumDriftDetector()
    >>> detector.fit(normal_data)  # level and scale inferred from normal data
    >>> anomalies = detector.detect(drifting_data)
    """

    def __init__(
        self, slack: float = 0.5, threshold: float = 5.0, direction: str = "both"
    ):
        super().__init__()

        if slack < 0:
            raise InvalidArgumentError("slack", "non-negative")
        if threshold <= 0:
            raise InvalidArgumentError("threshold", "positive")

        self._slack = slack
        self._threshold = threshold

        valid_directions = ("both", "positive", "negative")
        if direction in valid_directions:
            self._direction = direction
        else:
            raise ValueError(
                f"Selected direction, '{direction}' is not a valid direction. Valid directions are: {valid_directions}"
            )

        self._center: float | None = None
        self._scale: float | None = None

    @property
    def slack(self) -> float:
        return self._slack

    @property
    def threshold(self) -> float:
        return self._threshold

    def _fit(self, data: pd.Series):
        """Set the expected level and scale from normal data."""
        values = data.to_numpy(dtype=float)

        if not np.isfinite(values).any():
            raise ValueError("Input data contains no valid values")

        center = np.nanmedian(values)
        scale = GAUSSIAN_SCALE_FACTOR * np.nanmedian(np.abs(values - center))

        if not np.isfinite(center):
            raise ValueError("Input data contains no valid values")
        if not scale > 0:
            raise ValueError(
                "Could not determine a scale from the training data, because more "
                "than half of it is a single constant value. Consider "
                "ConstantValueDetector for such data."
            )

        self._center = float(center)
        self._scale = float(scale)
        return self

    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._center is None or self._scale is None:
            raise NotFittedError(
                tip="The expected level and scale of the normal data are needed "
                "to accumulate deviations."
            )

        if data.empty:
            return data.astype(bool)

        detect_positive = self._direction in ("both", "positive")
        detect_negative = self._direction in ("both", "negative")

        standardized = (data - self._center) / self._scale
        return standardized.apply(
            lambda col: _cusum(
                col.to_numpy(dtype=float),
                self._slack,
                self._threshold,
                detect_positive,
                detect_negative,
            ),
            axis=0,
        ).astype(bool)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(slack:{self._slack}, "
            f"threshold:{self._threshold}, direction:{self._direction})"
        )
