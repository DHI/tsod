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

STEP_RATIO = 0.25
"""Largest relative deviation of a step from the median step for the index to
count as evenly spaced: every step must lie strictly between
``(1 - STEP_RATIO)`` and ``(1 + STEP_RATIO)`` times the median step.

Detectors that need evenly spaced data count points, so what matters is that
each step covers one sampling period. Jitter within this band leaves that
intact; a missing or repeated sample (a step of about 2 or 0 times the median)
does not. The band is relative, so the same rule holds at any sampling rate and
for every index type."""


def _is_time_index(index: pd.Index) -> bool:
    return isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex))


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

    def _has_supported_index(self) -> bool:
        """Whether the index is datetime, timedelta or numeric."""
        index = self.frame.index
        if _is_time_index(index):
            return True
        return pd.api.types.is_numeric_dtype(
            index.dtype
        ) and not pd.api.types.is_bool_dtype(index.dtype)

    def _is_sorted(self) -> bool:
        """Whether the index is sorted in increasing order."""
        return self.frame.index.is_monotonic_increasing

    def _is_evenly_spaced(self) -> bool:
        """Whether every step of the index is within `STEP_RATIO` of the median step.

        Checks spacing only, not order: a decreasing index with equal steps is
        evenly spaced.
        """
        index = self.frame.index
        if len(index) < 3:
            return True
        values = index.asi8 if _is_time_index(index) else index.to_numpy(float)
        steps = np.diff(values)
        median = np.median(steps)
        if median == 0:
            return False
        return bool(np.all(np.abs(steps / median - 1) < STEP_RATIO))


class SortedData(SeriesData):
    """Data whose index is sorted in increasing order (duplicates allowed)."""


class EvenlySpacedData(SortedData):
    """Sorted data whose index steps are equal, within `STEP_RATIO`."""


def categorize(frame: pd.DataFrame) -> SeriesData:
    """Return `frame` wrapped in the most specific category it satisfies.

    Each property is checked at most once, from general to specific.

    Raises
    ------
    UnsupportedIndexError
        If the index is not a datetime, timedelta or numeric index.
    """
    data = SeriesData(frame)
    if not data._has_supported_index():
        raise UnsupportedIndexError(
            "tsod requires a DatetimeIndex, TimedeltaIndex or numeric index. "
            f"Got {type(frame.index).__name__} with dtype {frame.index.dtype} instead."
        )
    if not data._is_sorted():
        return data
    if not data._is_evenly_spaced():
        return SortedData(frame)
    return EvenlySpacedData(frame)
