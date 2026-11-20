import numpy as np
import pandas as pd
import pytest

from tsod.data import (
    STEP_RATIO,
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


# These tests follow STEP_RATIO, so changing its value needs no change here.
def hourly_with_last_step_scaled_by(ratio):
    index = pd.date_range("2020", periods=4, freq="1h")
    last = index[-2] + ratio * pd.Timedelta(hours=1)
    return index[:-1].append(pd.DatetimeIndex([last]))


@pytest.mark.parametrize("ratio", [1 - STEP_RATIO / 2, 1 + STEP_RATIO / 2])
def test_step_within_step_ratio_of_the_median_is_even(ratio):
    index = hourly_with_last_step_scaled_by(ratio)
    assert type(categorize(frame(index))) is EvenlySpacedData


@pytest.mark.parametrize("ratio", [1 - 2 * STEP_RATIO, 1 + 2 * STEP_RATIO])
def test_step_beyond_step_ratio_of_the_median_is_uneven(ratio):
    index = hourly_with_last_step_scaled_by(ratio)
    assert type(categorize(frame(index))) is SortedData


def test_missing_sample_is_uneven():
    index = pd.Index([0.0, 1.0, 2.0, 4.0, 5.0])
    assert type(categorize(frame(index))) is SortedData


def test_step_ratio_is_independent_of_sampling_rate():
    jittered = np.array([0.0, 1.0, 2.1, 3.0, 4.0])
    daily = pd.Timestamp("2020") + pd.to_timedelta(jittered, unit="D")
    secondly = pd.Timestamp("2020") + pd.to_timedelta(jittered, unit="s")
    assert type(categorize(frame(daily))) is EvenlySpacedData
    assert type(categorize(frame(secondly))) is EvenlySpacedData


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
