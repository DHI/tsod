from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from tsod.custom_exceptions import WrongInputDataType, NoRangeDefinedError, NonUniqueTimeStamps


class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    def __init__(self):
        pass

    def fit(self, data: pd.Series):
        """ Set detector parameters based on data. """
        self.validate(data)
        return self

    @abstractmethod
    def detect(self, data: pd.Series):
        "Detect anomalies"
        NotImplementedError()

    def validate(self, data):
        if not isinstance(data, pd.Series):
            raise WrongInputDataType()

    def _gradient(self, data: pd.Series):
        dt = data.index.to_series().diff().dt.total_seconds()
        if dt.min() < 1e-15:
            raise ValueError("Input must be monotonic increasing")

        gradient = data.diff() / dt
        return gradient

    def __str__(self):
        return f"{self.__class__.__name__}"


class AnomalyDetectionPipeline(BaseDetector):
    """ Combine detectors. 

    It is possible to combine several anomaly detection strategies into a combined detector.
    
    Examples
    --------
    >>> anomaly_detector = AnomalyDetectionPipeline([RangeDetector(), DiffRangeDetector()])
    >>> anomaly_detector.fit(normal_data)
    >>> detected_anomalies = anomaly_detector.detect(abnormal_data)
    """

    def __init__(self, detectors):
        super().__init__()
        self._detectors = detectors
        self._series_name = "is_anomaly"

    def fit(self, data):
        for detector in self._detectors:
            detector.fit(data)
        return self

    def detect(self, potentially_abnormal_data):
        detected_anomalies = [False] * len(potentially_abnormal_data)
        for detector in self._detectors:
            detected_anomalies |= detector.detect(potentially_abnormal_data)

        return detected_anomalies

    def detect_detailed(self, potentially_abnormal_data):
        detected_anomalies = pd.DataFrame(index=potentially_abnormal_data.index)
        detected_anomalies[self._series_name] = False
        for detector in self._detectors:
            name = str(detector)
            detected_anomalies[name] = detector.detect(potentially_abnormal_data)
            detected_anomalies[self._series_name] |= detected_anomalies[name]

        return detected_anomalies


class RangeDetector(BaseDetector):
    """ Detect values outside range. """
    def __init__(self, min_value=-np.inf, max_value=np.inf, quantiles=None):
        """ Set min or max manually. Optionally change quantiles used in fit().

        Parameters
        ----------
        min_value : float
            Minimum value threshold.
        max_value : float
            Maximum value threshold.
        quantiles : list[2]
                    Default quantiles [0, 1]. Same as min and max value.

        Examples
        --------
        >>> detector = RangeDetector(min_value=0.0, max_value=2.0)
        >>> anomalies = detector.detect(data)
        >>>
        >>> detector = RangeDetector()
        >>> detector.fit(normal_data) # min, max inferred from normal data
        >>> anomalies = detector.detect(data)
        >>>
        >>> detector = RangeDetector(quantiles=[0.001,0.999])
        >>> detector.fit(normal_data_with_some_outliers)
        >>> anomalies = detector.detect(data)
        """

        super().__init__()
        
        self._min = min_value
        
        self._max = max_value

        if quantiles is None:
            self._quantiles = [0.0, 1.0]
        else:
            assert 0.0 <= quantiles[0] <= 1.0
            assert 0.0 <= quantiles[1] <= 1.0
            self._quantiles = quantiles

    def fit(self, data):
        """ Set min and max based on data.

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

    def detect(self, data):
        "Detect anomalies"

        super().validate(data)
        self._validate_fit()

        if self._max is None:
            return data < self._min

        if self._min is None:
            return data > self._max

        return (data < self._min) | (data > self._max)

    def _validate_fit(self):
        if self._min is None and self._max is None:
            raise NoRangeDefinedError()

    def __str__(self):

        return f"{super.__str__(self)}{self._min}, {self._max})"

    def __repr__(self):
        return f"{self.__class__.__name__}(min: {self._min:.1e}, max: {self._max:.1e})"


class DiffRangeDetector(RangeDetector):
    """ Detect values outside diff or rate of change. """

    def __init__(self, min_value=None, max_value=None, time_unit='s'):
        super().__init__(min_value, max_value)
        if not time_unit == 's':
            raise Exception("Can currently only handle diff ranges per seconds")
        self._time_unit = time_unit

    def diff_time_series(self, data):
        # TODO handle non-equidistant data
        time_diff = data.index.shift() - data.index
        if any(time_diff == 0):
            raise NonUniqueTimeStamps()

        return data.diff() / time_diff.total_seconds()

    def fit(self, data):
        super().validate(data)
        data_diff = self.diff_time_series(data)

        self._min = data_diff.min() if self._min is None else self._min
        self._max = data_diff.max() if self._max is None else self._max
        return self

    def detect(self, data):
        "Detect anomalies"
        return super().detect(data.diff())


class RollingStandardDeviationDetector(BaseDetector):
    def __init__(self, window_size=10, threshold=0.1):
        super().__init__()
        self._window_size = window_size
        self._threshold = threshold

    def fit(self, data):
        super().validate(data)
        return self

    def detect(self, data):
        super().validate(data)
        anomalies = data.rolling(self._window_size).std() > self._threshold
        anomalies = anomalies.astype(int).diff() > 0  # only take positive edges
        anomalies[0] = False  # first element cannot be determined by diff
        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"


class ConstantValueDetector(BaseDetector):
    """
    Detect constant values over a longer period.

    Commonly caused by sensor failures, which get stuck at a constant level.
    """
    def __init__(self, window_size: int = 5, threshold: float = 1e-7):
        super().__init__()
        self._threshold = threshold
        self._window_size = window_size

    def fit(self, data):
        super().validate(data)
        return self

    def detect(self, data):
        super().validate(data)
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
    """
    def __init__(self, window_size: int = 5):
        super().__init__(window_size=window_size)

    def detect(self, data):
        super().validate(data)
        gradient = self._gradient(data)
        return super().detect(gradient)

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size})"
