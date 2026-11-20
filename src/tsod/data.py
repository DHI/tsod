"""Data categories passed from `Detector` to `_fit()` and `_detect()`.

`Detector.fit()` and `Detector.detect()` sort the input into the most specific
category with `categorize()`. Each detector declares the category it needs in
the type hint of the `data` argument of `_fit()` and `_detect()`; the base
class reads that hint and checks the data against it.

Categories form a hierarchy, so data of a more specific category is valid
wherever a more general one is accepted::

    SeriesData            datetime, timedelta or numeric index
    └── SortedData        index sorted in increasing order
        └── EvenlySpacedData  steps between index values are equal

Construct these classes only through `categorize()`, which checks the
properties each category promises.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .custom_exceptions import UnsupportedIndexError

# TODO: consult the spacing tolerance with domain experts.
SPACING_TOLERANCE = pd.Timedelta(seconds=1)
"""Largest deviation from the median step for a time index to count as evenly
spaced. A numeric index must have equal steps within floating point precision."""


@dataclass(frozen=True, eq=False)
class SeriesData:
    """Data with a datetime, timedelta or numeric index, in any order."""

    frame: pd.DataFrame

    @property
    def series(self) -> pd.Series:
        """The single column of `frame` as a Series."""
        if self.frame.shape[1] != 1:
            raise ValueError("Input DataFrame must contain exactly one column.")
        return self.frame.iloc[:, 0]


class SortedData(SeriesData):
    """Data whose index is sorted in increasing order (duplicates allowed)."""


class EvenlySpacedData(SortedData):
    """Sorted data whose index steps are equal, within `SPACING_TOLERANCE`."""


def _is_time_index(index: pd.Index) -> bool:
    return isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex))


def _check_index_type(index: pd.Index) -> None:
    """Raise `UnsupportedIndexError` unless the index is datetime, timedelta or numeric."""
    if _is_time_index(index):
        return
    if pd.api.types.is_numeric_dtype(index.dtype) and not pd.api.types.is_bool_dtype(
        index.dtype
    ):
        return
    raise UnsupportedIndexError(
        "tsod requires a DatetimeIndex, TimedeltaIndex or numeric index. "
        f"Got {type(index).__name__} with dtype {index.dtype} instead."
    )


def _is_evenly_spaced(index: pd.Index) -> bool:
    """Whether every step of a sorted index is within tolerance of the median step."""
    if len(index) < 3:
        return True
    steps = index[1:] - index[:-1]
    if _is_time_index(index):
        return abs(steps - steps.median()).max() <= SPACING_TOLERANCE
    steps = np.asarray(steps, dtype=float)
    return bool(np.allclose(steps, np.median(steps)))


def categorize(frame: pd.DataFrame) -> SeriesData:
    """Return `frame` wrapped in the most specific category it satisfies.

    Raises
    ------
    UnsupportedIndexError
        If the index is not a datetime, timedelta or numeric index.
    """
    index = frame.index
    _check_index_type(index)
    if not index.is_monotonic_increasing:
        return SeriesData(frame)
    if not _is_evenly_spaced(index):
        return SortedData(frame)
    return EvenlySpacedData(frame)
