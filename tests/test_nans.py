import numpy as np
import pandas as pd
import pytest

from tsod import (
    CombinedDetector,
    ConstantValueDetector,
    DiffDetector,
    HampelDetector,
    RangeDetector,
)


def test_detect_omits_nan_rows():
    data = pd.Series([0.0, np.nan, 10.0, 10.0])

    with pytest.warns(UserWarning, match="2 slices"):
        anomalies = RangeDetector(max_value=5.0).detect(data)

    assert anomalies.tolist() == [False, False, True, True]


def test_detect_never_compares_across_a_nan():
    data = pd.Series([0.0, 0.0, np.nan, 10.0, 10.0])

    with pytest.warns(UserWarning):
        anomalies = DiffDetector(max_diff=5.0).detect(data)

    assert not anomalies.any()


def test_detect_runs_on_each_slice():
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=30), rng.normal(size=30) + 20.0
    first[[3, 28]] = 10.0
    data = pd.Series(np.concatenate([first, [np.nan], second]))
    detector = HampelDetector(window_size=4)

    with pytest.warns(UserWarning):
        anomalies = detector.detect(data)

    expected = [
        *detector.detect(pd.Series(first)),
        False,
        *detector.detect(pd.Series(second)),
    ]
    assert anomalies.tolist() == expected
    assert anomalies.iloc[28]


def test_detect_keeps_the_input_index():
    index = pd.date_range("2020", periods=5, freq="1h")
    data = pd.Series([1.0, np.nan, 1.0, 1.0, 1.0], index=index)

    with pytest.warns(UserWarning):
        anomalies = ConstantValueDetector(window_size=3).detect(data)

    assert anomalies.index.equals(index)
    assert anomalies.dtype == bool
    assert anomalies.tolist() == [False, False, True, True, True]


def test_detect_handles_nan_per_column():
    df = pd.DataFrame(
        {"with_nan": [0.0, np.nan, 10.0, 10.0], "without_nan": [0.0, 0.0, 10.0, 10.0]}
    )

    with pytest.warns(UserWarning):
        anomalies = DiffDetector(max_diff=5.0).detect(df)

    assert anomalies["with_nan"].tolist() == [False, False, False, False]
    assert anomalies["without_nan"].tolist() == [False, False, True, False]


def test_detect_all_nan_returns_no_anomalies():
    data = pd.Series([np.nan, np.nan, np.nan])

    with pytest.warns(UserWarning):
        anomalies = RangeDetector(max_value=5.0).detect(data)

    assert anomalies.tolist() == [False, False, False]


def test_detect_without_nan_does_not_warn(recwarn):
    RangeDetector(max_value=5.0).detect(pd.Series([1.0, 2.0, 3.0]))

    assert len(recwarn) == 0


def test_fit_omits_nan_rows():
    data = pd.Series([0.0, np.nan, 4.0])

    with pytest.warns(UserWarning):
        detector = RangeDetector().fit(data)

    assert (detector._min, detector._max) == (0.0, 4.0)


def test_fit_all_nan_raises():
    with pytest.raises(ValueError, match="all NaN"):
        RangeDetector().fit(pd.Series([np.nan, np.nan]))


def test_combined_detector_leaves_nans_to_its_detectors():
    data = pd.Series([0.0, np.nan, 10.0, 10.0])
    combined = CombinedDetector(
        [RangeDetector(max_value=5.0), DiffDetector(max_diff=5.0)]
    )

    with pytest.warns(UserWarning) as record:
        anomalies = combined.detect(data)

    assert len(record) == 2
    assert anomalies.tolist() == [False, False, True, True]
