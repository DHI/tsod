import pytest
import numpy as np
import pandas as pd

from anomalydetection.custom_exceptions import NoRangeDefinedError, WrongInputDataType
from anomalydetection.detectors import (
    RangeDetector,
    DiffRangeDetector,
    AnomalyDetectionPipeline,
    RollingStandardDeviationDetector,
    ConstantValueDetector,
    ConstantGradientDetector)
from anomalydetection.features import create_dataset
from anomalydetection.hampel import HampelDetector
from anomalydetection.autoencoders import AutoEncoder
from anomalydetection.autoencoder_lstm import AutoEncoderLSTM

from tests.data_generation import create_random_walk_with_outliers


@pytest.fixture
def data_series():
    n_steps = 100
    (
        time_series_with_outliers,
        outlier_indices,
        random_walk,
    ) = create_random_walk_with_outliers(n_steps)
    time = pd.date_range(start="2020", periods=n_steps, freq="1H")
    return (
        pd.Series(time_series_with_outliers, index=time),
        outlier_indices,
        pd.Series(random_walk, index=time),
    )


@pytest.fixture
def range_data():
    normal_data = np.array([0, np.nan, 1, 0, 2, np.nan, 3.14, 4])
    abnormal_data = np.array([-1, np.nan, 2, np.nan, 1, 0, 4, 10])
    expected_anomalies = np.array([True, False, False, False, False, False, True, True])
    assert len(expected_anomalies) == len(abnormal_data)
    return normal_data, abnormal_data, expected_anomalies


@pytest.fixture
def range_data_series(range_data):
    normal_data, abnormal_data, expected_anomalies = range_data
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1H")
    return (
        pd.Series(normal_data, index=time),
        pd.Series(abnormal_data, index=time),
        expected_anomalies,
    )


@pytest.fixture
def constant_gradient_data_series(range_data):
    normal_data = np.array([0, np.nan, 1, 1.1, 1.4, 1.5555, 3.14, 4])
    abnormal_data = np.array([-1, 2.0, 2.1, 2.2, 2.3, 2.4, 4, 10])
    expected_anomalies = np.array([False, False, True, True, True, False, False, False])
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1H")
    return (
        pd.Series(normal_data, index=time),
        pd.Series(abnormal_data, index=time),
        expected_anomalies,
    )


@pytest.fixture
def constant_data_series(range_data):
    normal_data = np.array([0, np.nan, 1, 1.1, 1.4, 1.5555, 3.14, 4])
    abnormal_data = np.array([-1, np.nan, 1, 1, 1, 1, 4, 10])
    expected_anomalies = np.array(
        [False, False, False, True, True, False, False, False]
    )
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1H")
    return (
        pd.Series(normal_data, index=time),
        pd.Series(abnormal_data, index=time),
        expected_anomalies,
    )


def test_base_detector_exceptions(range_data, range_data_series):
    data, _, _ = range_data
    data_series, _, _ = range_data_series

    detector = RangeDetector()
    pytest.raises(NoRangeDefinedError, detector.detect, data_series)
    pytest.raises(WrongInputDataType, detector.fit, data)


def test_range_detector(range_data_series):
    data, _, _ = range_data_series

    detector = RangeDetector(0, 2)
    anomalies = detector.detect(data)
    expected_anomalies = [False, False, False, False, False, False, True, True]
    assert len(anomalies) == len(data)
    assert sum(anomalies) == 2
    assert all(expected_anomalies == anomalies)


def test_range_detector_autoset(range_data_series):
    data, _, _ = range_data_series

    anomalies = RangeDetector(min_value=3).detect(data)
    assert sum(anomalies) == 4

    anomalies = RangeDetector(max_value=3).detect(data)
    assert sum(anomalies) == 2


def test_range_detector_quantile():
    np.random.seed(42)
    train = np.random.normal(size=1000)
    test = np.random.normal(size=1000)

    train[42] = -6.5
    train[560] = 10.5

    test[142] = -4.5
    test[960] = 5.5

    normal_data_incl_two_outliers = pd.Series(train)
    test_data = pd.Series(test)

    # all test data is within range of train data, no anomalies detected
    nqdetector = RangeDetector().fit(normal_data_incl_two_outliers)
    detected_anomalies = nqdetector.detect(test_data)
    assert sum(detected_anomalies) == 0

    # exclude extreme values
    detector = RangeDetector(quantiles=[0.001, 0.999]).fit(normal_data_incl_two_outliers)
    detected_anomalies = detector.detect(test_data)
    assert sum(detected_anomalies) == 2
    assert detector._min > normal_data_incl_two_outliers.min()
    assert detector._max < normal_data_incl_two_outliers.max()


def test_diff_range_detector_autoset(range_data_series):
    normal_data, abnormal_data, expected_anomalies = range_data_series

    detector = DiffRangeDetector().fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    assert sum(detected_anomalies) == 3


def test_range_detector_pipeline(range_data_series):
    normal_data, abnormal_data, expected_anomalies = range_data_series
    anomaly_detector = AnomalyDetectionPipeline([RangeDetector(), DiffRangeDetector()])

    expected_anomalies[-3] = True  # Set diff range expected anomaly
    anomaly_detector.fit(normal_data)
    detected_anomalies = anomaly_detector.detect(abnormal_data)
    assert all(detected_anomalies == expected_anomalies)

    detected_anomalies = anomaly_detector.detect_detailed(abnormal_data)
    assert all(detected_anomalies.is_anomaly == expected_anomalies)


def test_rollingstddev_detector(range_data_series):
    data, _, _ = range_data_series

    detector = RollingStandardDeviationDetector(3, 0.1)
    anomalies = detector.detect(data)

    assert len(anomalies) == len(data)
    assert sum(anomalies) == 1


def test_hampel_detector(data_series):
    data_with_anomalies, expected_anomalies_indices, _ = data_series
    detector = HampelDetector()
    anomalies = detector.detect(data_with_anomalies)
    anomalies_indices = np.array(np.where(anomalies)).flatten()
    # Validate if the found anomalies are also in the expected anomaly set
    # NB Not necessarily all of them
    assert all(i in expected_anomalies_indices for i in anomalies_indices)

    detector_numba = HampelDetector(use_numba=True)
    anomalies_numba = detector_numba.detect(data_with_anomalies)
    anomalies_indices_numba = np.array(np.where(anomalies_numba)).flatten()
    assert all(i in anomalies_indices_numba for i in anomalies_indices)


def test_autoencoder_detector(data_series):
    data_with_anomalies, expected_anomalies_indices, normal_data = data_series
    detector = AutoEncoder(hidden_neurons=[1, 1, 1, 1], epochs=1)  # TODO add lagged features to increase layer size
    detector.fit(normal_data)
    anomalies = detector.detect(data_with_anomalies)
    anomalies_indices = np.array(np.where(anomalies)).flatten()
    # Validate if the found anomalies are also in the expected anomaly set
    # NB Not necessarily all of them
    # assert all(i in expected_anomalies_indices for i in anomalies_indices)


def test_autoencoderlstm_detector(data_series):
    data_with_anomalies, expected_anomalies_indices, normal_data = data_series
    detector = AutoEncoderLSTM()
    detector.fit(data_with_anomalies)
    anomalies = detector.detect(data_with_anomalies)
    anomalies_indices = np.array(np.where(anomalies)).flatten()


def test_constant_value_detector(constant_data_series):
    good_data, abnormal_data, _ = constant_data_series

    detector = ConstantValueDetector(2, 0.0001)
    anomalies = detector.detect(good_data)

    assert len(anomalies) == len(good_data)
    assert sum(anomalies) == 0

    detector = ConstantValueDetector(3, 0.0001)
    anomalies = detector.detect(abnormal_data)

    assert len(anomalies) == len(abnormal_data)
    assert sum(anomalies) == 2


def test_constant_gradient_detector(constant_gradient_data_series):
    good_data, abnormal_data, _ = constant_gradient_data_series

    detector = ConstantGradientDetector(3)
    anomalies = detector.detect(good_data)

    assert len(anomalies) == len(good_data)
    assert sum(anomalies) == 0

    detector = ConstantGradientDetector(3)
    anomalies = detector.detect(abnormal_data)

    assert len(anomalies) == len(abnormal_data)
    assert sum(anomalies) == 2
