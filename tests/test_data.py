import numpy as np
import pandas as pd
import pytest

from tsod.custom_exceptions import UnsupportedIndexError
from tsod.data import (
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
        pd.RangeIndex(5),
        pd.Index([0.0, 0.5, 1.0, 1.5]),
        pd.date_range("2020", periods=5, freq="10min"),
        pd.date_range("2020", periods=5, freq="10min", tz="Europe/Copenhagen"),
        pd.timedelta_range(0, periods=5, freq="1s"),
    ],
)
def test_evenly_spaced_index_is_evenly_spaced_data(index):
    assert type(categorize(frame(index))) is EvenlySpacedData


def time_index(*seconds):
    return pd.Timestamp("2020") + pd.to_timedelta(seconds, unit="s")


def test_time_steps_within_one_second_of_the_median_are_even():
    index = time_index(0, 60, 120, 180.9)
    assert type(categorize(frame(index))) is EvenlySpacedData


def test_time_steps_more_than_one_second_from_the_median_are_uneven():
    index = time_index(0, 60, 120, 182)
    assert type(categorize(frame(index))) is SortedData


def test_missing_timestamp_makes_data_uneven():
    index = pd.date_range("2020", periods=6, freq="1h").delete(3)
    assert type(categorize(frame(index))) is SortedData


def test_uneven_numeric_index_is_sorted_data():
    assert type(categorize(frame(pd.Index([0, 1, 2, 4])))) is SortedData


def test_duplicate_timestamps_are_sorted_but_uneven():
    index = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    )
    assert type(categorize(frame(index))) is SortedData


def test_unsorted_index_is_series_data():
    assert type(categorize(frame(pd.Index([2, 0, 1])))) is SeriesData


@pytest.mark.parametrize("n", [0, 1, 2])
def test_short_data_is_evenly_spaced(n):
    index = pd.date_range("2020", periods=n, freq="1h")
    assert type(categorize(frame(index))) is EvenlySpacedData


@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["a", "b", "c"]),
        pd.Index([True, False, True]),
        pd.period_range("2020", periods=3, freq="D"),
    ],
)
def test_non_numeric_index_raises(index):
    with pytest.raises(UnsupportedIndexError, match="numeric index"):
        categorize(frame(index))


def test_categories_are_a_hierarchy():
    assert issubclass(EvenlySpacedData, SortedData)
    assert issubclass(SortedData, SeriesData)


def test_series_returns_single_column():
    data = categorize(frame(pd.RangeIndex(3)))
    pd.testing.assert_series_equal(data.series, data.frame["x"])


def test_series_raises_for_several_columns():
    data = categorize(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    with pytest.raises(ValueError, match="exactly one column"):
        data.series
