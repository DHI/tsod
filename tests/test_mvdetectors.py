import pytest
import pandas as pd
import numpy as np

from tsod.mvdetectors import MVRangeDetector


@pytest.fixture
def range_data():
    n_obs = 15
    normal_data = pd.DataFrame(np.random.uniform(size=[3, n_obs]))
    normal_data.iloc[2, [2, 8]] = np.nan
    abnormal_data = pd.DataFrame(np.random.uniform(size=[3, n_obs]))
    abnormal_data.iloc[0, [2, 3, 7]] = 5
    abnormal_data.iloc[1, [2, 12]] = -2
    abnormal_data.iloc[0, [8]] = np.nan
    abnormal_data.iloc[2, [8, 9]] = np.nan
    return normal_data, abnormal_data


@pytest.mark.parametrize("detector, expected_anomalies_list", [
    (MVRangeDetector(min_value=0.0, max_value=1.0),
     [False, False, True, True, False, False, False, True, False, False, False, False, True, False, False]),
    (MVRangeDetector(max_value=1.0),
     [False, False, True, True, False, False, False, True, False, False, False, False, False, False, False]),
    (MVRangeDetector(min_value=0.0),
     [False, False, True, False, False, False, False, False, False, False, False, False, True, False, False]),
])
def test_range_detector_detection(range_data, detector, expected_anomalies_list):
    normal_data, abnormal_data = range_data
    n_obs = normal_data.shape[1]
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.Series(expected_anomalies_list, index=pd.Int64Index(np.arange(n_obs), dtype='int64'))
    pd.testing.assert_series_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not any(detected_anomalies)


def test_range_detector_fitting(range_data):
    normal_data, abnormal_data = range_data
    detector = MVRangeDetector()
    detector.fit(normal_data)
    n_obs = normal_data.shape[1]
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.Series(
        [False, False, True, True, False, False, False, True, False, False, False, False, True, False, False],
        index=pd.Int64Index(np.arange(n_obs), dtype='int64'))
    pd.testing.assert_series_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not any(detected_anomalies)
