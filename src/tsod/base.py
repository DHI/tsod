from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, overload
from typing_extensions import Self

from pathlib import Path
import warnings
import joblib

import numpy as np
import pandas as pd

from .custom_exceptions import NaNInDataError, WrongInputDataTypeError


class NanPolicy(str, Enum):
    """How a detector handles NaN values in the input data.

    RAISE
        Raise `NaNInDataError` if the data contains NaN.
    OMIT
        Warn, drop the NaN rows and run on each slice of consecutive data between
        them. In the output of `detect`, the dropped rows are False.
    """

    RAISE = "raise"
    OMIT = "omit"


def _nan_policy(nans: Union[str, NanPolicy]) -> NanPolicy:
    try:
        return NanPolicy(nans)
    except ValueError:
        valid = [policy.value for policy in NanPolicy]
        raise ValueError(f"nans must be one of {valid}, got {nans!r}") from None


def _consecutive_slices(isna: np.ndarray) -> list[np.ndarray]:
    """Positions of the slices of consecutive non-NaN values."""
    positions = np.flatnonzero(~isna)
    if positions.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    return np.split(positions, breaks)


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
    """Base class for anomaly detectors.

    Parameters
    ----------
    nans : {'omit', 'raise'} or NanPolicy, default='omit'
        How to handle NaN values. 'omit' warns, drops the NaN rows and runs on
        each slice of consecutive data between them, so values on either side of
        a NaN are never compared. 'raise' raises `NaNInDataError`.
    """

    # Detectors that pass the data on to other detectors (CombinedDetector)
    # leave NaN handling to them.
    _delegates_nans: bool = False

    def __init__(self, nans: Union[str, NanPolicy] = NanPolicy.OMIT):
        self._nans = _nan_policy(nans)

    @property
    def nans(self) -> NanPolicy:
        # Detectors saved before NaN handling existed have no _nans
        return getattr(self, "_nans", NanPolicy.OMIT)

    def _check_nans(self, n_nans: int, n_slices: int, stacklevel: int) -> None:
        message = f"Input data contains {n_nans} NaN values"
        if self.nans is NanPolicy.RAISE:
            raise NaNInDataError(f"{message}. Pass nans='omit' to skip them.")
        warnings.warn(
            f"{message}; they were omitted and {self} ran on {n_slices} slices of "
            "consecutive data. Pass nans='raise' to raise instead.",
            UserWarning,
            stacklevel=stacklevel + 1,
        )

    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> Self:
        """Set detector parameters based on data.

        Parameters
        ----------
        data: pd.Series or pd.DataFrame
            Normal (non-anomalous) time series data for training.
            If DataFrame, must contain exactly one column
        Returns
        -------
        Self
        """
        df = self.validate(data)

        if df.empty:
            raise ValueError("Input data cannot be empty")
        if df.shape[1] != 1:
            raise ValueError("Input DataFrame must contain exactly one column.")

        series = df.iloc[:, 0]
        isna = series.isna().to_numpy()
        if self._delegates_nans or not isna.any():
            self._fit(series)
            return self

        slices = _consecutive_slices(isna)
        if not slices:
            raise ValueError("Input data cannot be all NaN")
        self._check_nans(int(isna.sum()), len(slices), stacklevel=2)
        self._fit_slices([series.iloc[positions] for positions in slices])
        return self

    def _fit(self, data: pd.Series):
        # Default implementation is a NoOp
        return self

    def _fit_slices(self, slices: list[pd.Series]):
        """Fit on slices of consecutive data without NaN.

        The default joins the slices and fits on the result, which is right for
        detectors that look at each point on its own. Detectors that compare
        neighbouring points override this to compute per slice.
        """
        return self._fit(pd.concat(slices))

    @overload
    def detect(self, data: pd.Series) -> pd.Series: ...

    @overload
    def detect(self, data: pd.DataFrame) -> pd.DataFrame: ...

    def detect(
        self, data: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Detect anomalies

        Parameters
        ----------
        data: pd.Series or pd.DataFrame
            Time series data with possible anomalies

        Returns
        -------
        pd.Series or pd.DataFrame
            Time series with bools, True == anomaly.
        """
        series_as_input = isinstance(data, pd.Series)
        data_as_dataframe = self.validate(data)

        pred = self._detect_omitting_nans(data_as_dataframe)

        if series_as_input:
            return pred.iloc[:, 0]

        return pred

    def _detect_omitting_nans(self, data: pd.DataFrame) -> pd.DataFrame:
        isna = data.isna().to_numpy()
        if self._delegates_nans or not isna.any():
            return self._detect(data)

        # Each column has its own NaN: a NaN in one column does not remove
        # the timestamp from the others
        slices_per_column = [
            _consecutive_slices(isna[:, j]) for j in range(data.shape[1])
        ]
        n_slices = sum(len(slices) for slices in slices_per_column)
        self._check_nans(int(isna.sum()), n_slices, stacklevel=3)

        result = np.zeros(data.shape, dtype=bool)
        for j, slices in enumerate(slices_per_column):
            for positions in slices:
                piece = self._detect(data.iloc[positions, [j]])
                result[positions, j] = piece.iloc[:, 0].to_numpy(dtype=bool)
        return pd.DataFrame(result, index=data.index, columns=data.columns)

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
