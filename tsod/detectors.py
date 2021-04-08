"""Simple univariate anomaly detectors"""

from collections.abc import Sequence
import pandas as pd
import numpy as np

from .base import Detector


class CombinedDetector(Detector, Sequence):
    """Combine detectors.

    It is possible to combine several anomaly detection strategies into a combined detector.

    Examples
    --------
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

    def _fit(self, data):
        for detector in self._detectors:
            detector.fit(data)
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        all_anomalies = []
        for detector in self._detectors:
            anom = detector.detect(data)
            all_anomalies.append(anom)
        data_frame = pd.DataFrame(all_anomalies).T
        return data_frame.any(axis=1)

    def __getitem__(self, index):
        return self._detectors[index]

    def __len__(self):
        return len(self._detectors)


class RangeDetector(Detector):
    """
    Detect values outside range.

    Parameters
    ----------
    min_value : float
        Minimum value threshold.
    max_value : float
        Maximum value threshold.
    quantiles : list[2]
                Default quantiles [0, 1]. Same as min and max value.

    Examples
    ---------
    >>> detector = RangeDetector(min_value=0.0, max_value=2.0)
    >>> anomalies = detector.detect(data)

    >>> detector = RangeDetector()
    >>> detector.fit(normal_data) # min, max inferred from normal data
    >>> anomalies = detector.detect(data)

    >>> detector = RangeDetector(quantiles=[0.001,0.999])
    >>> detector.fit(normal_data_with_some_outliers)
    >>> anomalies = detector.detect(data)"""

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

    def _fit(self, data):
        """Set min and max based on data.

        Parameters
        ----------
        data :  pd.Series
                Normal time series data.
        """
        super().validate(data)

        quantiles = np.quantile(data.dropna(), self._quantiles)
        self._min = quantiles.min()
        self._max = quantiles.max()

        assert self._max >= self._min
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        "Detect anomalies outside range"

        if self._max is None:
            return data < self._min

        if self._min is None:
            return data > self._max

        return (data < self._min) | (data > self._max)

    def __str__(self):

        return f"{super.__str__(self)}{self._min}, {self._max})"

    def __repr__(self):
        return f"{self.__class__.__name__}(min: {self._min:.1e}, max: {self._max:.1e})"


class DiffDetector(Detector):
    """Detect sudden shifts in data. Irrespective of time axis.

    Parameters
    ----------
    max_diff : float
        Maximum change threshold.
    direction: str
        positive, negative or both, default='both'

    See also
    --------
    GradientDetector: similar functionality but considers actual time between data points
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

    def _fit(self, data):
        data_diff = data.diff()

        self._max_diff = data_diff.max()
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        if self._direction == "both":
            return np.abs(data.diff()) > self._max_diff
        elif self._direction == "positive":
            return data.diff() > self._max_diff
        else:
            return data.diff() < -self._max_diff

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self._max_diff}, direction:{self._direction})"
        )


class RollingStandardDeviationDetector(Detector):
    """Detect large variations


    ----------
    window_size: int
        Number of data points to evaluate over, default=10
    max_std: float
        Maximum standard deviation to accept as normal, default=np.inf
    center: bool
        Center rolling window, default=True
    """

    def __init__(self, window_size=10, max_std=np.inf, center=True):
        super().__init__()
        self._window_size = window_size
        self._max_std = max_std
        self._center = center

    def _fit(self, data):
        self._max_std = data.rolling(self._window_size).std().max()

        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        anomalies = (
            data.rolling(self._window_size, center=self._center).std() > self._max_std
        )
        # anomalies = anomalies.astype(int).diff() > 0  # only take positive edges
        anomalies[0] = False  # first element cannot be determined by diff
        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}(window_size:{self._window_size}, max_std:{self._max_std})"


class ConstantValueDetector(Detector):
    """
    Detect constant values over a longer period.

    Commonly caused by sensor failures, which get stuck at a constant level.
    """

    def __init__(self, window_size: int = 3, threshold: float = 1e-7):
        super().__init__()
        self._threshold = threshold
        self._window_size = window_size

    def _fit(self, data):
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        rollmax = data.rolling(self._window_size, center=True).apply(np.nanmax)
        rollmin = data.rolling(self._window_size, center=True).apply(np.nanmin)
        anomalies = np.abs(rollmax - rollmin) < self._threshold
        anomalies[0] = False  # first element cannot be determined
        anomalies[-1] = False
        idx = np.where(anomalies)[0]
        if idx is not None:
            # assuming window size = 3
            # remove also points before and after each detected anomaly
            anomalies[idx[idx > 0] - 1] = True
            maxidx = len(anomalies) - 1
            anomalies[idx[idx < maxidx] + 1] = True

        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"


class ConstantGradientDetector(ConstantValueDetector):
    """Detect constant gradients.

    Typically caused by linear interpolation over a long interval.

    Parameters
    ==========
    window_size: int
        Minium window to consider as anomaly, default 3
    """

    def __init__(self, window_size: int = 3):
        super().__init__(window_size=window_size)

    def _detect(self, data: pd.Series) -> pd.Series:
        gradient = self._gradient(data, periods=1)
        s1 = super()._detect(gradient)
        gradient = self._gradient(data, periods=-1)
        s2 = super()._detect(gradient)
        return s1 | s2

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size})"


class GradientDetector(Detector):
    """Detects abrupt changes

    Parameters
    ==========
    max_gradient: float
        Maximum rate of change per second, default np.inf
    direction: str
        positive, negative or both, default='both'
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
        """ Set max gradient based on data. """

        self._max_gradient = np.max(np.abs(self._gradient(data)))
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        gradient = self._gradient(data)
        if self._direction == "negative":
            return gradient < -self._max_gradient
        elif self._direction == "positive":
            return gradient > self._max_gradient
        else:
            return np.abs(gradient) > self._max_gradient

    def __str__(self):
        max_grad_hr = self._max_gradient * 3600.0
        return (
            f"{self.__class__.__name__}({max_grad_hr}/hr, direction:{self._direction})"
        )
