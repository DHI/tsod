from tsod.base import Detector
import pytest
import numpy as np
import pandas as pd

from tsod.custom_exceptions import WrongInputDataType
from tsod.detectors import (
    RangeDetector,
    DiffDetector,
    CombinedDetector,
    RollingStandardDeviationDetector,
    ConstantValueDetector,
    ConstantGradientDetector,
    GradientDetector,
)

from tsod.features import create_dataset
from tsod.hampel import HampelDetector
from tsod.autoencoders import AutoEncoder
from tsod.autoencoder_lstm import AutoEncoderLSTM

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
    abnormal_data = np.array([-1.0, np.nan, 2.0, np.nan, 1.0, 0.0, 4.1, 10.0])
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
    expected_anomalies = np.array([False, True, True, True, True, True, False, False])
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
    expected_anomalies = np.array([False, False, True, True, True, True, False, False])
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


def test_combined_fit(range_data_series):
    normal_data, abnormal_data, labels = range_data_series
    cd = CombinedDetector([ConstantValueDetector(), RangeDetector()])
    cd.fit(normal_data)

    anomalies = cd.detect(abnormal_data)
    assert all(anomalies == labels)


def test_combined_wrong_type():
    with pytest.raises(ValueError):
        CombinedDetector([ConstantValueDetector, RangeDetector()])  #


def test_combined_access_items():

    cd = CombinedDetector([ConstantValueDetector(), RangeDetector()])

    assert isinstance(cd[0], Detector)
    assert isinstance(cd[0], ConstantValueDetector)
    assert isinstance(cd[1], RangeDetector)
    assert isinstance(cd[-1], RangeDetector)


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
    detector = RangeDetector(quantiles=[0.001, 0.999]).fit(
        normal_data_incl_two_outliers
    )
    detected_anomalies = detector.detect(test_data)
    assert sum(detected_anomalies) == 2
    assert detector._min > normal_data_incl_two_outliers.min()
    assert detector._max < normal_data_incl_two_outliers.max()


def test_diff_detector_autoset(range_data_series):
    normal_data, abnormal_data, expected_anomalies = range_data_series

    detector = DiffDetector().fit(normal_data)
    detected_anomalies = detector.detect(abnormal_data)
    assert sum(detected_anomalies) == 2


def test_combined_detector():
    df = pd.read_csv("tests/data/example.csv", parse_dates=True, index_col=0)
    combined = CombinedDetector(
        [
            ConstantValueDetector(),
            RangeDetector(max_value=2.0),
        ]
    )

    series = df.value
    res = combined.detect(series)

    assert isinstance(res, pd.Series)


def test_rollingstddev_detector():

    np.random.seed(42)
    normal_data = pd.Series(np.random.normal(scale=1.0, size=1000)) + 10.0 * np.sin(
        np.linspace(0, 10, num=1000)
    )
    abnormal_data = pd.Series(np.random.normal(scale=2.0, size=100))

    all_data = pd.concat([normal_data, abnormal_data])

    detector = RollingStandardDeviationDetector()
    anomalies = detector.detect(normal_data)
    assert sum(anomalies) == 0

    detector.fit(normal_data)
    anomalies = detector.detect(normal_data)
    assert sum(anomalies) == 0

    anomalies = detector.detect(all_data)
    assert sum(anomalies) > 0

    # Manual specification
    detector = RollingStandardDeviationDetector(max_std=2.0)
    anomalies = detector.detect(normal_data)
    assert sum(anomalies) == 0

    anomalies = detector.detect(all_data)
    assert sum(anomalies) > 0


def test_hampel_detector(data_series):
    data_with_anomalies, expected_anomalies_indices, _ = data_series
    detector = HampelDetector()
    anomalies = detector.detect(data_with_anomalies)
    anomalies_indices = np.array(np.where(anomalies)).flatten()
    # Validate if the found anomalies are also in the expected anomaly set
    # NB Not necessarily all of them
    assert all(i in expected_anomalies_indices for i in anomalies_indices)


def test_autoencoder_detector(data_series):
    data_with_anomalies, expected_anomalies_indices, normal_data = data_series
    detector = AutoEncoder(
        hidden_neurons=[1, 1, 1, 1], epochs=1
    )  # TODO add lagged features to increase layer size
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
    assert sum(anomalies) == 4


def test_constant_gradient_detector(constant_gradient_data_series):
    good_data, abnormal_data, _ = constant_gradient_data_series

    detector = ConstantGradientDetector(3)
    anomalies = detector.detect(good_data)

    assert len(anomalies) == len(good_data)
    assert sum(anomalies) == 0

    detector = ConstantGradientDetector(3)
    anomalies = detector.detect(abnormal_data)

    assert len(anomalies) == len(abnormal_data)
    assert sum(anomalies) == 5


def test_gradient_detector_constant_gradient(constant_gradient_data_series):
    good_data, _, _ = constant_gradient_data_series

    detector = GradientDetector(1.0)
    anomalies = detector.detect(good_data)

    assert len(anomalies) == len(good_data)
    assert sum(anomalies) == 0


def test_gradient_detector_sudden_jump():

    normal_data = np.array(
        [
            -0.5,
            -0.6,
            0.6,
            0.6,
            0.1,
            0.6,
            0.4,
            0.8,
            0.7,
            1.5,
            1.6,
            1.1,
            0.3,
            2.1,
            0.7,
            0.3,
            -1.7,
            -0.3,
            0.0,
            -1.0,
        ]
    )
    abnormal_data = np.array(
        [
            -0.5,
            -1.5,
            1.5,
            0.6,
            0.1,
            0.6,
            0.4,
            0.8,
            0.7,
            1.5,
            1.6,
            1.1,
            0.3,
            2.1,
            0.7,
            0.3,
            -1.7,
            -0.3,
            0.0,
            -1.0,
        ]
    )

    expected_anomalies = np.repeat(False, len(normal_data))
    expected_anomalies[2] = True
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1H")

    normal_data = pd.Series(normal_data, index=time)
    abnormal_data = pd.Series(abnormal_data, index=time)

    detector = GradientDetector()

    anomalies = detector.detect(normal_data)
    assert sum(anomalies) == 0

    # Default is to accept any gradient
    anomalies = detector.detect(abnormal_data)
    assert sum(anomalies) == 0

    # Max gradient 2.0/h
    detector.fit(normal_data)
    anomalies = detector.detect(abnormal_data)

    assert sum(anomalies) == 1


def test_create_dataset(data_series):
    data_with_anomalies, _, _ = data_series
    data_with_anomalies.name = "y"
    data = data_with_anomalies.to_frame()
    time_steps = 2
    X, y = create_dataset(data[["y"]], data.y, time_steps)
    assert len(y) == len(data) - time_steps
    assert X.shape[0] == len(data) - time_steps
    assert X.shape[1] == time_steps
