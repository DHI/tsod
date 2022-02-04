import pytest
import pandas as pd
import numpy as np

from tsod.custom_exceptions import InvalidArgumentError
from tsod.mvdetectors import MVRangeDetector


@pytest.fixture
def range_data():
    n_obs = 15
    normal_data = pd.DataFrame(np.random.uniform(size=[3, n_obs]))
    normal_data.iloc[2, [2, 8]] = np.nan
    normal_data.iloc[:, 13] = 1
    normal_data.iloc[:, 14] = 0
    abnormal_data = pd.DataFrame(np.random.uniform(size=[3, n_obs]))
    abnormal_data.iloc[0, [2, 3, 7]] = 5
    abnormal_data.iloc[1, [2, 12]] = -2
    abnormal_data.iloc[0, [8]] = np.nan
    abnormal_data.iloc[2, [8, 9]] = np.nan
    return normal_data, abnormal_data


@pytest.fixture
def range_data_time_series_specific_ranges():
    n_obs = 15
    ts_mins = [-1, -0.5, 0]
    ts_maxs = [2, 3, 4]
    normal_data = pd.DataFrame(np.random.uniform(low=ts_mins, high=ts_maxs, size=(n_obs, len(ts_mins))).T)
    normal_data.iloc[2, [2, 8]] = np.nan
    normal_data.iloc[:, 13] = ts_mins
    normal_data.iloc[:, 14] = ts_maxs
    abnormal_data = pd.DataFrame(np.random.uniform(low=ts_mins, high=ts_maxs, size=(n_obs, len(ts_mins))).T)
    abnormal_data.iloc[0, [2, 3, 7]] = 5
    abnormal_data.iloc[1, [2, 12]] = -2
    abnormal_data.iloc[0, [8]] = np.nan
    abnormal_data.iloc[2, [8, 9]] = np.nan
    return normal_data, abnormal_data


@pytest.mark.parametrize("detector, expected_anomalies_list", [
    (MVRangeDetector(min_value=0.0, max_value=1.0),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]),
    (MVRangeDetector(max_value=1.0),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]),
    (MVRangeDetector(min_value=0.0),
     [[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]])
])
def test_single_range_detector_detection(range_data, detector, expected_anomalies_list):
    normal_data, abnormal_data = range_data
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(expected_anomalies_list, columns=abnormal_data.columns, index=abnormal_data.index)
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


def test_single_range_detector_fitting(range_data):
    normal_data, abnormal_data = range_data
    detector = MVRangeDetector()
    detector.fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
         [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]],
        columns=abnormal_data.columns, index=abnormal_data.index)
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize("detector, expected_anomalies_list", [
    (MVRangeDetector(min_value=[0.0, 0.0, 0.0], max_value=1.0),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]),
    (MVRangeDetector(min_value=0.0, max_value=[1.0, 1.0, 1.0]),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]),
    (MVRangeDetector(min_value=[0.0, 0.0, 0.0], max_value=[1.0, 1.0, 1.0]),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]])
])
def test_multi_range_detector_detection(range_data, detector, expected_anomalies_list):
    normal_data, abnormal_data = range_data
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(expected_anomalies_list, columns=abnormal_data.columns, index=abnormal_data.index)
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)

    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize("detector, expected_anomalies_list", [
    (MVRangeDetector(min_value=[-1, -0.5, 0], max_value=[2, 3, 4]),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]),
    (MVRangeDetector(max_value=[2, 3, 4]),
     [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]]),
    (MVRangeDetector(min_value=[-1, -0.5, 0]),
     [[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
     [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]])
])
def test_multiple_ranges_detector_detection(range_data_time_series_specific_ranges, detector, expected_anomalies_list):
    normal_data, abnormal_data = range_data_time_series_specific_ranges
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(expected_anomalies_list, columns=abnormal_data.columns, index=abnormal_data.index)
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


def test_multiple_ranges_detector_fitting(range_data_time_series_specific_ranges):
    normal_data, abnormal_data = range_data_time_series_specific_ranges
    detector = MVRangeDetector()
    detector.fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        [[False, False, True, True, False, False, False, True, False, False, False, False, False, False, False],
         [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False],
         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]],
        columns=abnormal_data.columns, index=abnormal_data.index)
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize("min_value, max_value",
                         [
                             (3, 2), ([0, 0, 3], 2), ([[0], [0], [0]], 1), (-1, [[0], [0], [0]])
                         ])
def test_invalid_argument_raised_min_max(min_value, max_value):
    with pytest.raises(InvalidArgumentError):
        MVRangeDetector(min_value=min_value, max_value=max_value)


@pytest.mark.parametrize("quantile_prob_cut_offs",
                         [
                             ([0.5, 1.1]), ([-0.5, 1.1]), ([-0.5, 0.9])
                         ])
def test_invalid_argument_raised_quantiles(quantile_prob_cut_offs):
    with pytest.raises(InvalidArgumentError):
        MVRangeDetector(quantiles=quantile_prob_cut_offs)
