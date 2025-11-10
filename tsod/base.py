from abc import ABC, abstractmethod
from typing import Union

from pathlib import Path
import joblib

import pandas as pd


from .custom_exceptions import WrongInputDataTypeError
detectDataType = Union[pd.Series, pd.DataFrame]

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
        if not (isinstance(data, pd.Series)):
            raise WrongInputDataTypeError()
        
        self._fit(data)
        return self

    def _fit(self, data: pd.Series):
        # Default implementation is a NoOp
        return self

    def detect(self, data: detectDataType) -> detectDataType:
        """Detect anomalies

        Parameters
        ----------
        data: pd.Series or pd.DataFrame
                Time series data with possible anomalies

        Returns
        -------
        pd.Series or pd.DataFrame
            Time series with bools, True == anomaly
        """
        data = self.validate(data)

        pred = self._detect(data)
        return self._postprocess(pred)

    def _postprocess(self, pred: detectDataType) -> detectDataType:
        # TODO implement
        return pred

    @abstractmethod
    def _detect(self, data: detectDataType) -> detectDataType:
        """Detect anomalies"""
        pass

    def validate(self, data: Union[pd.Series, pd.DataFrame]):
        """
        Validate input data

        Parameters
        ----------
        data: pd.Series or pd.DataFrame
            Time series data
        Returns
        -------
        pd.Series or pd.DataFrame
            Validated time series data
        """
        
        if isinstance(data, pd.DataFrame):
            # check unique column names
            if not data.columns.is_unique:
                raise ValueError(
                    "DataFrame columns must be unique."
                )
        elif isinstance(data, pd.Series):
            pass
        else:
            raise WrongInputDataTypeError(
                "Input data must be a pandas.Series or pandas.DataFrame."
            )

        return data

    def _gradient(
        self, data: Union[pd.Series, pd.DataFrame], periods: int = 1
    ) -> Union[pd.Series, pd.DataFrame]:
        dt = data.index.to_series().diff().dt.total_seconds()
        if dt.min() < 1e-15:
            raise ValueError("Index must be monotonically increasing")

        # Broadcast division with dataframe correctly
        if isinstance(data, pd.DataFrame):
            gradient = data.diff(periods=periods).div(dt, axis=0)
        elif isinstance(data, pd.Series):
            gradient = data.diff(periods=periods)/dt
        else:
            raise WrongInputDataTypeError(
                "Input data must be a pandas.Series or pandas.DataFrame."
            )
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
