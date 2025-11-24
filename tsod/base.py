from abc import ABC, abstractmethod
from typing import Union, overload

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
    
    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> "Detector":
        """Set detector parameters based on data.

        Parameters
        ----------
        data: pd.Series or pd.DataFrame
            Normal (non-anomalous) time series data for training.
            If DataFrame, must contain exactly one column
        Returns
        -------
        Detector
            Self  
        """
        df = self._validate(data)

        if df.shape[1] != 1:
            raise ValueError("Input DataFrame must contain exactly one column.")
        
        self._fit(df.iloc[:,0])
        return self

    def _fit(self, data: pd.Series):
        # Default implementation is a NoOp
        return self
    
    @overload
    def detect(self, data: pd.Series) -> pd.Series: ...

    @overload
    def detect(self, data: pd.DataFrame) -> pd.DataFrame: ...

    def detect(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
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
        series_as_input = isinstance(data, pd.Series)
        data_as_dataframe = self._validate(data)

        pred = self._detect(data_as_dataframe)
        pred = self._postprocess(pred)
    
        if series_as_input:
            pred = pred.iloc[:,0]
             
        return pred

    def _postprocess(self, pred: pd.DataFrame) -> pd.DataFrame:
        # TODO implement
        return pred


    @abstractmethod
    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies"""
        pass


    def _validate(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and normalize input data.
        
        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Time series data
        
        Returns
        -------
        pd.DataFrame
            Validated and normalized data
        """
        # Check type
        if isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise WrongInputDataTypeError()
        
        # Check unique column names
        if not df.columns.is_unique:
            raise ValueError("DataFrame columns names must be unique.")
        
        if df.empty:
            raise ValueError("Input data cannot be empty")
          
        return df

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
