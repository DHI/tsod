import pandas as pd

from anomalydetection.custom_exceptions import WrongInputDataType, NoRangeDefinedError
from anomalydetection import hampel


class BaseDetector:
    def __init__(self):
        pass

    def fit(self, data: pd.Series):
        """ Set detector parameters based on data. """
        self.validate(data)
        return self

    def detect(self, data: pd.Series):
        NotImplementedError()

    def validate(self, data):
        if not isinstance(data, pd.Series):
            raise WrongInputDataType()


class AnomalyDetectionPipeline(BaseDetector):
    """ Combine detectors. """
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
    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self._min = min_value
        self._max = max_value

    def fit(self, data):
        """ Set min and max based on data.

        Parameters
        ----------
        data :  pd.Series
                Normal time series data.
        """
        super().validate(data)
        self._min = data.min() if self._min is None else self._min
        self._max = data.max() if self._max is None else self._max

        assert self._max >= self._min
        return self

    def detect(self, data):
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
        return f"{self.__class__.__name__}({self._min}, {self._max})"


class DiffRangeDetector(RangeDetector):
    """ Detect values outside diff or rate of change. """
    def fit(self, data):
        super().validate(data)
        data_diff = data.diff()
        self._min = data_diff.min() if self._min is None else self._min
        self._max = data_diff.max() if self._max is None else self._max
        return self

    def detect(self, data):
        return super().detect(data.diff())


class PeakDetector(BaseDetector):
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


class HampelDetector(BaseDetector):
    def __init__(self, window_size=5, threshold=3, use_numba=False):
        super().__init__()
        hampel.validate_arguments(window_size, threshold)
        self._threshold = threshold
        self._window_size = window_size
        self._use_numba = use_numba

    def detect(self, data):
        super().validate(data)

        if self._use_numba:
            anomalies, indices, _ = hampel.detect_using_numba(data.values, self._window_size, self._threshold)
        else:
            anomalies, indices, _ = hampel.detect(data, self._window_size, self._threshold)

        return anomalies

    def __str__(self):
        return f"{self.__class__.__name__}({self._window_size}, {self._threshold})"
