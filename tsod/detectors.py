from collections.abc import Sequence
import pandas as pd
import numpy as np

from .base import Detector
from .custom_exceptions import WrongInputDataType


class CombinedDetector(Detector, Sequence):
    """Combine detectors.

    It is possible to combine several anomaly detection strategies into a combined detector.

    Examples
    --------
    >>> anomaly_detector = CombinedDetector([RangeDetector(), DiffRangeDetector()])
    >>> anomaly_detector.fit(normal_data)
    >>> detected_anomalies = anomaly_detector.detect(abnormal_data)
    """

    def __init__(self, detectors):
        super().__init__()

        for d in detectors:
            if not isinstance(d, Detector):
                raise ValueError(
                    f"{d} is not a Detector. Did you forget to create an instance, e.g. ConstantValueDetector()?"
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
        df = pd.DataFrame(all_anomalies).T
        return df.any(axis=1)

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


class DiffRangeDetector(RangeDetector):
    """ Detect values outside diff or rate of change. """

    def __init__(self, min_value=None, max_value=None):
        super().__init__(min_value, max_value)

    def _diff_time_series(self, data):
        # TODO handle non-equidistant data
        time_diff = data.index.shift() - data.index

        return data.diff()

    def _fit(self, data):
        data_diff = self._diff_time_series(data)

        self._min = data_diff.min() if self._min is None else self._min
        self._max = data_diff.max() if self._max is None else self._max
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        return super()._detect(data.diff())


class RollingStandardDeviationDetector(Detector):
    def __init__(self, window_size=10, threshold=0.1):
        super().__init__()
        self._window_size = window_size
        self._threshold = threshold

    def _fit(self, data):

        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        anomalies = data.rolling(self._window_size).std() > self._threshold
        anomalies = anomalies.astype(int).diff() > 0  # only take positive edges
        anomalies[0] = False  # first element cannot be determined by diff
        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"


class ConstantValueDetector(Detector):
    """
    Detect constant values over a longer period.

    Commonly caused by sensor failures, which get stuck at a constant level.
    """

    def __init__(self, window_size: int = 5, threshold: float = 1e-7):
        super().__init__()
        self._threshold = threshold
        self._window_size = window_size

    def _fit(self, data):
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        rollmax = data.rolling(self._window_size).apply(np.nanmax)
        rollmin = data.rolling(self._window_size).apply(np.nanmin)
        anomalies = np.abs(rollmax - rollmin) < self._threshold
        anomalies[0] = False  # first element cannot be determined
        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"


class ConstantGradientDetector(ConstantValueDetector):
    """Detect constant gradients.

    Typically caused by linear interpolation over a long interval.

    Parameters
    ==========
    window_size: int
        Minium window to consider as anomaly, default 5
    """

    def __init__(self, window_size: int = 5):
        super().__init__(window_size=window_size)

    def _detect(self, data: pd.Series) -> pd.Series:
        gradient = self._gradient(data)
        return super()._detect(gradient)

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size})"


class MaxAbsGradientDetector(Detector):
    """Detects abrupt changes

    Parameters
    ==========
    max_abs_gradient: float
        Maximum rate of change per second, default np.inf
    """

    def __init__(self, max_abs_gradient=np.inf):
        self._max_abs_gradient = max_abs_gradient

    def _fit(self, data: pd.Series):
        """ Set max absolute gradient based on data. """

        self._max_abs_gradient = np.max(np.abs(self._gradient(data)))
        return self

    def _detect(self, data: pd.Series) -> pd.Series:
        gradient = self._gradient(data)
        return np.abs(gradient) > self._max_abs_gradient

    def __str__(self):
        max_grad_hr = self._max_abs_gradient * 3600.0
        return f"{self.__class__.__name__}({max_grad_hr:.3f}/hr)"
