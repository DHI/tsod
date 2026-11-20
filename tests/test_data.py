import numpy as np
import pandas as pd
import pytest

from tsod.data import (
    SPACING_TOLERANCE,
    EvenlySpacedData,
    SeriesData,
    SortedData,
    categorize,
)


def frame(index):
    return pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([0.0, 0.5, 1.0, 1.5]),
        pd.timedelta_range(0, periods=5, freq="1s"),
        pd.date_range("2020", periods=5, freq="10min", tz="Europe/Copenhagen"),
    ],
    ids=["float", "timedelta", "timezone-aware"],
)
def test_evenly_spaced_index_of_any_supported_type(index):
    assert type(categorize(frame(index))) is EvenlySpacedData


def test_duplicate_timestamps_are_sorted_but_uneven():
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    )
    assert type(categorize(frame(index))) is SortedData


# The spacing rule is pending agreement with domain experts. These tests follow
# SPACING_TOLERANCE, so changing its value needs no change here; changing the
# rule itself (e.g. what the deviation is measured against) does.
def hourly_with_last_step_off_by(deviation):
    index = pd.date_range("2020", periods=4, freq="1h")
    return index[:-1].append(pd.DatetimeIndex([index[-1] + deviation]))


def test_step_off_by_the_spacing_tolerance_is_even():
    index = hourly_with_last_step_off_by(SPACING_TOLERANCE)
    assert type(categorize(frame(index))) is EvenlySpacedData


def test_step_off_by_more_than_the_spacing_tolerance_is_uneven():
    index = hourly_with_last_step_off_by(2 * SPACING_TOLERANCE)
    assert type(categorize(frame(index))) is SortedData


def test_categories_are_a_hierarchy():
    assert issubclass(EvenlySpacedData, SortedData)
    assert issubclass(SortedData, SeriesData)


def test_series_returns_single_column():
    data = SeriesData(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    pd.testing.assert_series_equal(data.series, data.frame["x"])


def test_series_raises_for_several_columns():
    data = SeriesData(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    with pytest.raises(ValueError, match="exactly one column"):
        data.series
