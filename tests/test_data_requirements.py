import warnings

import numpy as np
import pandas as pd
import pytest

from tsod import (
    CombinedDetector,
    ConstantGradientDetector,
    ConstantValueDetector,
    DiffDetector,
    GradientDetector,
    HampelDetector,
    RangeDetector,
    RollingStandardDeviationDetector,
)
from tsod.base import Detector
from tsod.custom_exceptions import (
    UnevenSpacingWarning,
    UnsortedIndexError,
    UnsupportedIndexError,
)
from tsod.data import EvenlySpacedData, SeriesData, SortedData


class Recorder(Detector):
    """Records what _fit and _detect received."""

    def _fit(self, data):
        self.fitted_with = data
        return self

    def _detect(self, data):
        self.detected_with = data
        return data.frame < 0 if isinstance(data, SeriesData) else data < 0


class AcceptsAny(Recorder):
    def _fit(self, data: SeriesData):
        return super()._fit(data)

    def _detect(self, data: SeriesData) -> pd.DataFrame:
        return super()._detect(data)


class NeedsSorted(Recorder):
    def _fit(self, data: SortedData):
        return super()._fit(data)

    def _detect(self, data: SortedData) -> pd.DataFrame:
        return super()._detect(data)


class NeedsEvenSpacing(Recorder):
    def _fit(self, data: EvenlySpacedData):
        return super()._fit(data)

    def _detect(self, data: EvenlySpacedData) -> pd.DataFrame:
        return super()._detect(data)


even = pd.Series(
    [1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020", periods=4, freq="1h"), name="x"
)
uneven = even.drop(even.index[1])
unsorted = even.iloc[[1, 0, 2, 3]]


def test_detector_without_hints_gets_pandas():
    detector = Recorder().fit(even)
    detector.detect(even)
    assert isinstance(detector.fitted_with, pd.Series)
    assert isinstance(detector.detected_with, pd.DataFrame)


def test_detector_gets_most_specific_category():
    detector = AcceptsAny().fit(even)
    detector.detect(even)
    assert type(detector.fitted_with) is EvenlySpacedData
    assert type(detector.detected_with) is EvenlySpacedData


def test_any_order_accepted_when_series_data_declared():
    detector = AcceptsAny()
    detector.detect(unsorted)
    assert type(detector.detected_with) is SeriesData


@pytest.mark.parametrize("method", ["fit", "detect"])
def test_unsorted_data_raises_when_sorted_data_declared(method):
    with pytest.raises(UnsortedIndexError, match="sort_index"):
        getattr(NeedsSorted(), method)(unsorted)


@pytest.mark.parametrize("method", ["fit", "detect"])
def test_uneven_data_warns_when_even_spacing_declared(method):
    detector = NeedsEvenSpacing()
    with pytest.warns(UnevenSpacingWarning, match="NeedsEvenSpacing") as record:
        getattr(detector, method)(uneven)
    assert record[0].filename == __file__


def test_uneven_data_still_runs_after_warning():
    with pytest.warns(UnevenSpacingWarning):
        result = NeedsEvenSpacing().detect(uneven)
    pd.testing.assert_series_equal(result, uneven < 0)


def test_uneven_data_accepted_silently_when_sorted_data_declared():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NeedsSorted().detect(uneven)


@pytest.mark.parametrize("detector", [Recorder(), AcceptsAny(), NeedsEvenSpacing()])
def test_non_numeric_index_raises_for_every_detector(detector):
    data = pd.Series(np.arange(3.0), index=["a", "b", "c"])
    with pytest.raises(UnsupportedIndexError):
        detector.detect(data)


# Requirements of the tsod detectors

rng = np.random.default_rng(0)
values = rng.normal(size=30)
time = pd.date_range("2020", periods=30, freq="10min")
even_data = pd.Series(values, index=time, name="x")
uneven_data = even_data.drop(time[[5, 6, 20]])
unsorted_data = even_data.iloc[::-1]

EVEN_SPACING = [
    DiffDetector(),
    RollingStandardDeviationDetector(window_size=3),
    ConstantValueDetector(),
    HampelDetector(window_size=2),
]
ANY_SPACING = [GradientDetector(), ConstantGradientDetector()]


def name(detector):
    return type(detector).__name__


@pytest.mark.parametrize("detector", EVEN_SPACING, ids=name)
def test_point_counting_detectors_warn_on_uneven_data(detector):
    with pytest.warns(UnevenSpacingWarning, match=name(detector)):
        detector.detect(uneven_data)


@pytest.mark.parametrize("detector", ANY_SPACING + [RangeDetector()], ids=name)
def test_detectors_using_time_or_values_only_accept_uneven_data(detector):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detector.detect(uneven_data)


@pytest.mark.parametrize("detector", EVEN_SPACING + ANY_SPACING, ids=name)
def test_neighbour_comparing_detectors_raise_on_unsorted_data(detector):
    with pytest.raises(UnsortedIndexError):
        detector.detect(unsorted_data)


def test_range_detector_accepts_unsorted_data():
    result = RangeDetector(max_value=1.0).detect(unsorted_data)
    pd.testing.assert_series_equal(result, unsorted_data > 1.0)


def test_combined_detector_leaves_requirements_to_its_detectors():
    combined = CombinedDetector([RangeDetector(), DiffDetector()])
    with pytest.raises(UnsortedIndexError, match="DiffDetector"):
        combined.detect(unsorted_data)
    with pytest.warns(UnevenSpacingWarning, match="DiffDetector"):
        combined.detect(uneven_data)


@pytest.mark.parametrize(
    "detector",
    [DiffDetector(), RollingStandardDeviationDetector(), GradientDetector()],
    ids=name,
)
def test_fit_checks_requirements_too(detector):
    with pytest.raises(UnsortedIndexError):
        detector.fit(unsorted_data)
