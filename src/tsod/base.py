from abc import ABC, abstractmethod
import functools
import typing
from typing import Union, overload
import warnings
from typing_extensions import Self

from pathlib import Path
import joblib

import pandas as pd

from .custom_exceptions import (
    UnevenSpacingWarning,
    UnsortedIndexError,
    WrongInputDataTypeError,
)
from .data import EvenlySpacedData, SeriesData, SortedData, categorize


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


@functools.lru_cache(maxsize=None)
def _accepted_data_type(method) -> type | None:
    """The `SeriesData` category a `_fit`/`_detect` method declares for `data`.

    Returns None when the hint is missing or is not a `SeriesData` category, e.g.
    a detector written before the categories existed, which takes plain pandas.
    """
    hint = typing.get_type_hints(method).get("data")
    if isinstance(hint, type) and issubclass(hint, SeriesData):
        return hint
    return None


class Detector(ABC):
    """Base class for detectors.

    Subclasses implement `_detect()` and optionally `_fit()`, and declare the
    data they need with the type hint of `data`: `SeriesData`, `SortedData` or
    `EvenlySpacedData` (see `tsod.data`). `fit()` and `detect()` categorize the
    input and check it against that hint: they raise `UnsortedIndexError` for
    unsorted data when sorted data is required, and warn with
    `UnevenSpacingWarning` when evenly spaced data is required but not given.
    """

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

        self._fit(self._prepare(df, "_fit"))
        return self

    def _fit(self, data: pd.Series):
        # Default implementation is a NoOp
        return self

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

        pred = self._detect(self._prepare(data_as_dataframe, "_detect"))

        if series_as_input:
            return pred.iloc[:, 0]

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

        return df

    def _prepare(self, df: pd.DataFrame, method_name: str):
        """Categorize `df` and check it against what `method_name` accepts."""
        data = categorize(df)
        accepted = _accepted_data_type(getattr(type(self), method_name))
        if accepted is None:
            return data.series if method_name == "_fit" else data.frame
        if isinstance(data, accepted):
            return data
        if issubclass(accepted, SortedData) and not isinstance(data, SortedData):
            raise UnsortedIndexError(
                f"{type(self).__name__} requires an index sorted in increasing "
                "order. Sort the data first, e.g. with data.sort_index()."
            )
        if issubclass(accepted, EvenlySpacedData):
            warnings.warn(
                f"{type(self).__name__} counts data points and assumes evenly "
                "spaced data, but the index steps differ. Results may be "
                "misleading; consider resampling the data first.",
                UnevenSpacingWarning,
                stacklevel=3,
            )
        return data

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
