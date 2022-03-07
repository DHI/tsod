import pytest
import pandas as pd
import numpy as np


from tsod.custom_exceptions import (
    InvalidArgumentError,
    WrongInputSizeError,
    NoRangeDefinedError,
)
from tsod.mvdetectors import MVRangeDetector, MVCorrelationDetector


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
    normal_data = pd.DataFrame(
        np.random.uniform(low=ts_mins, high=ts_maxs, size=(n_obs, len(ts_mins))).T
    )
    normal_data.iloc[2, [2, 8]] = np.nan
    normal_data.iloc[:, 13] = ts_mins
    normal_data.iloc[:, 14] = ts_maxs
    abnormal_data = pd.DataFrame(
        np.random.uniform(low=ts_mins, high=ts_maxs, size=(n_obs, len(ts_mins))).T
    )
    abnormal_data.iloc[0, [2, 3, 7]] = 5
    abnormal_data.iloc[1, [2, 12]] = -2
    abnormal_data.iloc[0, [8]] = np.nan
    abnormal_data.iloc[2, [8, 9]] = np.nan
    return normal_data, abnormal_data


@pytest.fixture
def subsequence_data():
    n_obs = 100
    normal_data = pd.DataFrame(np.random.normal(size=[3, n_obs], loc=2.0, scale=0.1))
    normal_data.iloc[0, :] = 3 * normal_data.iloc[1, :] + normal_data.iloc[2, :]
    abnormal_data = pd.DataFrame(np.random.normal(size=[3, n_obs], loc=2.0, scale=0.1))
    abnormal_data.iloc[0, :] = 3 * abnormal_data.iloc[1, :] + abnormal_data.iloc[2, :]
    abnormal_data.iloc[0, 20:35] = np.random.normal(size=[1, 15], loc=2.0, scale=0.1)
    return normal_data, abnormal_data


class TestMVCorrelationDetector:
    @pytest.fixture(autouse=True)
    def _setup(self, subsequence_data):
        self.normal_data, self.abnormal_data = subsequence_data

    def test_subsequence(self):
        detector = MVCorrelationDetector(window_size=15)
        detector.fit(self.normal_data)
        detected_anomalies = detector.detect(self.abnormal_data)
        detected_row_indices = np.where(detected_anomalies)[0]

        assert all([val in detected_row_indices for val in np.arange(20, 35)])


@pytest.mark.parametrize(
    "detector, expected_anomalies_list",
    [
        (
            MVRangeDetector(min_value=0.0, max_value=1.0),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(max_value=1.0),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(min_value=0.0),
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
    ],
)
def test_single_range_detector_detection(range_data, detector, expected_anomalies_list):
    normal_data, abnormal_data = range_data
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        expected_anomalies_list,
        columns=abnormal_data.columns,
        index=abnormal_data.index,
    )
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


def test_single_range_detector_fitting(range_data):
    normal_data, abnormal_data = range_data
    detector = MVRangeDetector()
    detector.fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        [
            [
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        ],
        columns=abnormal_data.columns,
        index=abnormal_data.index,
    )
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize(
    "detector, expected_anomalies_list",
    [
        (
            MVRangeDetector(min_value=[0.0, 0.0, 0.0], max_value=1.0),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(min_value=0.0, max_value=[1.0, 1.0, 1.0]),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(min_value=[0.0, 0.0, 0.0], max_value=[1.0, 1.0, 1.0]),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
    ],
)
def test_multi_range_detector_detection(range_data, detector, expected_anomalies_list):
    normal_data, abnormal_data = range_data
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        expected_anomalies_list,
        columns=abnormal_data.columns,
        index=abnormal_data.index,
    )
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)

    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize(
    "detector, expected_anomalies_list",
    [
        (
            MVRangeDetector(min_value=[-1, -0.5, 0], max_value=[2, 3, 4]),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(max_value=[2, 3, 4]),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(min_value=[-1, -0.5, 0]),
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(max_value=[2, 3, 4], min_value=None),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
        (
            MVRangeDetector(min_value=[-1, -0.5, 0], max_value=None),
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        ),
    ],
)
def test_multiple_ranges_detector_detection(
    range_data_time_series_specific_ranges, detector, expected_anomalies_list
):
    normal_data, abnormal_data = range_data_time_series_specific_ranges
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        expected_anomalies_list,
        columns=abnormal_data.columns,
        index=abnormal_data.index,
    )
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


def test_multiple_ranges_detector_fitting(range_data_time_series_specific_ranges):
    normal_data, abnormal_data = range_data_time_series_specific_ranges
    detector = MVRangeDetector()
    detector.fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        [
            [
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        ],
        columns=abnormal_data.columns,
        index=abnormal_data.index,
    )
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize(
    "min_value, max_value",
    [(3, 2), ([0, 0, 3], 2), ([[0], [0], [0]], 1), (-1, [[0], [0], [0]])],
)
def test_invalid_argument_raised_min_max(min_value, max_value):
    with pytest.raises(InvalidArgumentError):
        MVRangeDetector(min_value=min_value, max_value=max_value)


@pytest.mark.parametrize(
    "quantile_prob_cut_offs", [([0.5, 1.1]), ([-0.5, 1.1]), ([-0.5, 0.9])]
)
def test_invalid_argument_raised_quantiles(quantile_prob_cut_offs):
    with pytest.raises(InvalidArgumentError):
        MVRangeDetector(quantiles=quantile_prob_cut_offs)


def test_invalid_argument_raised_wrong_input(range_data):
    normal_data, abnormal_data = range_data
    detector = MVRangeDetector([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(WrongInputSizeError):
        detector.detect(normal_data)


@pytest.mark.parametrize(
    "detector, expected_anomalies_list",
    [
        (
            MVRangeDetector(quantiles=[0.0, 1.0]),
            [
                [
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ],
        )
    ],
)
def test_multiple_ranges_quantile_detector_detection(
    range_data_time_series_specific_ranges, detector, expected_anomalies_list
):
    normal_data, abnormal_data = range_data_time_series_specific_ranges
    detector.fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.DataFrame(
        expected_anomalies_list,
        columns=abnormal_data.columns,
        index=abnormal_data.index,
    )
    pd.testing.assert_frame_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


@pytest.mark.parametrize(
    "detector, expected_anomalies_list",
    [
        (
            MVRangeDetector(quantiles=[0.0, 1.0]),
            [
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    ],
)
def test_multiple_ranges_quantile_single_ts_detection(
    range_data_time_series_specific_ranges, detector, expected_anomalies_list
):
    normal_data, abnormal_data = range_data_time_series_specific_ranges
    normal_data = normal_data.iloc[0, :]
    abnormal_data = abnormal_data.iloc[0, :]

    detector.fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    expected_anomalies = pd.Series(
        expected_anomalies_list, index=abnormal_data.index, name=abnormal_data.name
    )
    pd.testing.assert_series_equal(expected_anomalies, detected_anomalies)

    detected_anomalies = detector.detect(normal_data)
    assert not detected_anomalies.to_numpy().any()


def test_no_range_defined_raised(range_data):
    normal_data, abnormal_data = range_data
    detector = MVRangeDetector(min_value=None, max_value=None)
    with pytest.raises(NoRangeDefinedError):
        detector.detect(normal_data)
