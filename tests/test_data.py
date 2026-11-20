import pandas as pd
import pytest

from tsod.data import EvenlySpacedData, SeriesData, SortedData


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
