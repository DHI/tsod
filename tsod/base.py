from abc import ABC, abstractmethod
from typing import Union

from pathlib import Path
import joblib

import pandas as pd


from .custom_exceptions import WrongInputDataTypeError


def load(path: Union[str, Path]):
    """Load a saved model from disk saved with `Detector.save`

    Parameters
    ==========
    path: str or Path
        file-like object to load detector from
    """

    return joblib.load(path)


class Detector(ABC):
    """Abstract base class for all detectors"""

    def __init__(self):
        pass

    def fit(self, data: pd.Series):
        """Set detector parameters based on data.

        Parameters
        ----------
        data:  pd.Series
                Normal time series data.
        """
        data = self.validate(data)
        self._fit(data)
        return self

    def _fit(self, data: pd.Series):
        # Default implementation is a NoOp
        return self

    def detect(self, data: pd.Series) -> pd.Series:
        """Detect anomalies

        Parameters
        ----------
        data: pd.Series
                Time series data with possible anomalies

        Returns
        -------
        pd.Series
            Time series with bools, True == anomaly
        """
        data = self.validate(data)

        pred = self._detect(data)
        return self._postprocess(pred)

    def _postprocess(self, pred: pd.Series) -> pd.Series:
        # TODO implement
        return pred

    @abstractmethod
    def _detect(self, data: pd.Series) -> pd.Series:
        """Detect anomalies"""
        pass

    def validate(
        self, data: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Check that input data is in correct format and possibly adjust"""
        if not (isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)):
            raise WrongInputDataTypeError()
        return data

    def _gradient(
        self, data: Union[pd.Series, pd.DataFrame], periods: int = 1
    ) -> pd.Series:
        dt = data.index.to_series().diff().dt.total_seconds()
        if dt.min() < 1e-15:
            raise ValueError("Index must be monotonically increasing")

        gradient = data.diff(periods=periods) / dt
        return gradient

    def __str__(self):
        return f"{self.__class__.__name__}"

    def save(self, path: Union[str, Path]) -> None:
        """Save a detector for later use

        Parameters
        ==========
        path: str or Path
            file-like object to load detector from
        """

        joblib.dump(self, path)
