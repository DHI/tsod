from abc import ABC, abstractmethod
from typing import Union, overload

from pathlib import Path
import joblib

import pandas as pd

from .custom_exceptions import WrongInputDataTypeError

def load(path: Union[str, Path]):
    """Load a saved model from disk saved with `Detector.save`

    Parameters
    ----------
    path : str or Path
        File-like object to load detector from.

    Returns
    -------
    Detector
        The loaded detector instance.
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
        df = self.validate(data)

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
        data_as_dataframe = self.validate(data)

        pred = self._detect(data_as_dataframe)
    
        if series_as_input:
            pred = pred.iloc[:,0]
             
        return pred

    @abstractmethod
    def _detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies"""
        pass


    def validate(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Check that input data is in correct format and possibly adjust.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input data to validate.

        Returns
        -------
        pd.DataFrame
            Validated data.

        Raises
        ------
        WrongInputDataTypeError
            If data is not a pd.Series or pd.DataFrame.
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
        """Save a detector for later use.

        Parameters
        ----------
        path : str or Path
            File path to save the detector to.
        """

        joblib.dump(self, path)
