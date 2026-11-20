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
from tsod.custom_exceptions import (
    UnevenSpacingWarning,
    UnsortedIndexError,
    UnsupportedIndexError,
)

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


def test_uneven_spacing_warning_points_at_the_caller():
    with pytest.warns(UnevenSpacingWarning) as record:
        DiffDetector().detect(uneven_data)
    assert record[0].filename == __file__


def test_detector_still_runs_after_uneven_spacing_warning():
    with pytest.warns(UnevenSpacingWarning):
        result = DiffDetector().detect(uneven_data)
    expected = DiffDetector().detect(uneven_data.reset_index(drop=True))
    np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())


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


@pytest.mark.parametrize("detector", [RangeDetector(), DiffDetector()], ids=name)
def test_non_numeric_index_raises(detector):
    data = pd.Series(np.arange(3.0), index=["a", "b", "c"])
    with pytest.raises(UnsupportedIndexError):
        detector.detect(data)


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
