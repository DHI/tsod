from tsod.base import Detector
import pytest
import numpy as np
import pandas as pd
import os

from tsod.custom_exceptions import WrongInputDataTypeError
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


from tests.data_generation import create_random_walk_with_outliers


@pytest.fixture
def data_series():
    n_steps = 100
    (
        time_series_with_outliers,
        outlier_indices,
        random_walk,
    ) = create_random_walk_with_outliers(n_steps)
    time = pd.date_range(start="2020", periods=n_steps, freq="1h")
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
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1h")
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
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1h")
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
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1h")
    return (
        pd.Series(normal_data, index=time),
        pd.Series(abnormal_data, index=time),
        expected_anomalies,
    )


def test_base_detector_exceptions(range_data, range_data_series):
    data, _, _ = range_data
    data_series, _, _ = range_data_series

    detector = RangeDetector()
    pytest.raises(WrongInputDataTypeError, detector.fit, data)


def test_range_detector(range_data_series):
    data, _, _ = range_data_series

    detector = RangeDetector(0, 2)
    anomalies = detector.detect(data)
    expected_anomalies = [False, False, False, False, False, False, True, True]
    assert isinstance(anomalies, pd.Series)
    assert len(anomalies) == len(data)
    assert sum(anomalies) == 2
    assert all(expected_anomalies == anomalies)

def test_range_detector_frame_1col(range_data_series):
    data, _, _ = range_data_series

    # One column dataframe
    detector = RangeDetector(0, 2)
    anomalies = detector.detect(data.to_frame())
    expected_anomalies = [False, False, False, False, False, False, True, True]

    assert isinstance(anomalies, pd.DataFrame)
    assert anomalies.shape == (len(data),1)
    assert (anomalies.iloc[:,0].sum() == 2)
    assert expected_anomalies == anomalies.iloc[:,0].values.tolist()


def test_range_detector_frame_multicol(range_data_series):
    data, _, _ = range_data_series
    # Multi column dataframe
    detector = RangeDetector(0, 2)
    df = pd.concat([data.rename("col1"), (data*2).rename("col2")], axis=1)
    anomalies = detector.detect(df)
    expected_anomalies = [
        [False, False, False, False, False, False, True, True],
        [False, False, False, False, True, False, True, True],
    ]

    assert isinstance(anomalies, pd.DataFrame)
    assert list(anomalies.columns) == ['col1', 'col2']
    assert anomalies.shape == (len(data),2)
    assert anomalies.iloc[:,0].sum() == 2
    assert anomalies.iloc[:,1].sum() == 3
    assert expected_anomalies == anomalies.T.values.tolist()

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

def test_combined_fit_frame(range_data_series):
    normal_data, abnormal_data, labels = range_data_series
    cd = CombinedDetector([ConstantValueDetector(), RangeDetector()])
    cd.fit(normal_data)
    df = pd.concat([normal_data.rename("col1"), (abnormal_data).rename("col2")], axis=1)
    anomalies = cd.detect(df)

    assert anomalies["col1"].tolist() == [False]*len(labels) # should be normal data
    assert anomalies["col2"].tolist() == labels.tolist()

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

def test_diff_detector_autoset_frame(range_data_series):
    normal_data, abnormal_data, expected_anomalies = range_data_series

    df = pd.concat(
        [normal_data.rename("normal"), abnormal_data.rename("abnormal")], axis=1
    )

    detector = DiffDetector().fit(normal_data)
    detected_anomalies = detector.detect(df)
    assert detected_anomalies["abnormal"].sum() == 2
    assert detected_anomalies["normal"].sum() == 0



def test_combined_detector():
    path_to_tests_super_folder = os.path.abspath(__file__).split("tests")[0]
    df = pd.read_csv(
        os.path.join(path_to_tests_super_folder, "tests", "data", "example.csv"),
        parse_dates=True,
        index_col=0,
    )
    combined = CombinedDetector(
        [
            ConstantValueDetector(),
            RangeDetector(max_value=2.0),
        ]
    )

    series = df.value
    res = combined.detect(series)
    assert isinstance(res, pd.Series)

    res = combined.detect(df)
    assert isinstance(res, pd.DataFrame)

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
    #anomalies_frame = detector.detect(normal_data.to_frame())
    #assert sum(anomalies_frame.iloc[:,0]) == 0

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

def test_rollingstddev_detector_frame():
    np.random.seed(42)
    normal_data = pd.Series(np.random.normal(scale=1.0, size=1000)) + 10.0 * np.sin(
        np.linspace(0, 10, num=1000)
    )
    abnormal_data = pd.Series(np.random.normal(scale=10000.0, size=1000))
    df = pd.concat([normal_data.rename("normal"), abnormal_data.rename("abnormal")], axis=1)

    detector = RollingStandardDeviationDetector()
    anomalies = detector.detect(df)
    assert anomalies.loc[:, "normal"].sum() == 0
    assert anomalies.loc[:, "abnormal"].sum() == 0

    detector.fit(normal_data)
    anomalies = detector.detect(df)
    assert anomalies.loc[:, "normal"].sum() == 0
    assert anomalies.loc[:, "abnormal"].sum() > 0

def test_hampel_detector(data_series):
    data_with_anomalies, expected_anomalies_indices, _ = data_series
    detector = HampelDetector()
    anomalies = detector.detect(data_with_anomalies)
    anomalies_indices = np.array(np.where(anomalies)).flatten()
    # Validate if the found anomalies are also in the expected anomaly set
    # NB Not necessarily all of them
    assert all(i in expected_anomalies_indices for i in anomalies_indices)


def test_hampel_detector_frame(data_series):
    data_with_anomalies, expected_anomalies_indices, _ = data_series
    data_with_anomalies_frame = pd.concat(
        [data_with_anomalies.rename("col1"), data_with_anomalies.rename("col2")], axis=1
    )
    detector = HampelDetector()
    anomalies = detector.detect(data_with_anomalies_frame)
    for col in anomalies.columns:
        anomalies_indices = np.array(np.where(anomalies[col])).flatten()
        assert all(i in expected_anomalies_indices for i in anomalies_indices)

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


def test_constant_gradient_detector_frame(constant_gradient_data_series):
    good_data, abnormal_data, _ = constant_gradient_data_series

    df = pd.concat(
        [good_data.rename("normal"), abnormal_data.rename("abnormal")], axis=1
    )
    detector = ConstantGradientDetector(3)
    anomalies = detector.detect(df)

    assert anomalies.shape == df.shape
    assert anomalies.loc[:,"normal"].sum() == 0
    assert anomalies.loc[:,"abnormal"].sum() == 5

def test_gradient_detector_constant_gradient(constant_gradient_data_series):
    good_data, _, _ = constant_gradient_data_series

    detector = GradientDetector(1.0)
    anomalies = detector.detect(good_data)

    assert len(anomalies) == len(good_data)
    assert sum(anomalies) == 0

def test_gradient_detector_constant_gradient_frame(constant_gradient_data_series):
    good_data, _, _ = constant_gradient_data_series

    detector = GradientDetector(1.0, direction="positive")
    df = pd.concat(
        [good_data.rename("normal"), good_data.rename("abnormal")], axis=1
    )
    df.iloc[3,1] = 5000  # introduce an anomaly in one column
    anomalies = detector.detect(df)

    assert anomalies.shape == df.shape
    assert anomalies.loc[:,"normal"].sum() == 0
    assert anomalies.loc[:,"abnormal"].sum() == 1
    assert anomalies.loc[df.index[3],"abnormal"] is np.True_


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
    time = pd.date_range(start="2020", periods=len(normal_data), freq="1h")

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


def test_gradient_detector_datetime_index_validation():
    """Test that GradientDetector raises ValueError when data doesn't have DatetimeIndex"""
    ###### Integer index test ######
    detector = GradientDetector()

    # Test data with integer index
    data_with_int_index = pd.Series([1, 2, 3, 4, 5])

    # Test with integer index
    with pytest.raises(ValueError, match="GradientDetector requires a DatetimeIndex"):
        detector.fit(data_with_int_index)

    ###### DatetimeIndex test ######
    detector = GradientDetector()
    
    # Test data with valid DatetimeIndex data works fine
    time = pd.date_range(start="2020", periods=5, freq="1h")
    data_with_datetime_index = pd.Series([1, 2, 3, 4, 5], index=time)

    # This should not raise an exception
    detector.fit(data_with_datetime_index)
    
def test_create_dataset(data_series):
    data_with_anomalies, _, _ = data_series
    data_with_anomalies.name = "y"
    data = data_with_anomalies.to_frame()
    time_steps = 2
    predictors, y = create_dataset(data[["y"]], data.y, time_steps)
    assert len(y) == len(data) - time_steps
    assert predictors.shape[0] == len(data) - time_steps
    assert predictors.shape[1] == time_steps


def test_gradient(constant_data_series):
    df = pd.Series([1, 1, 2, 1, 1], index=pd.date_range(start="2020", periods=5, freq="1min"))
    detector = RangeDetector()
    gradient = detector._gradient(df)
    assert type(gradient) is pd.Series
    assert len(gradient) == len(df)
    assert np.isnan(gradient.loc[df.index[0]])
    assert gradient.loc[df.index[1]] == 0
    assert gradient.loc[df.index[2]] == 1/60

def test_gradient_dataframe_1col(constant_data_series):
    df = pd.Series([1, 1, 2, 1, 1], index=pd.date_range(start="2020", periods=5, freq="1min"))
    detector = RangeDetector()
    gradient = detector._gradient(df.to_frame())
    assert type(gradient) is pd.DataFrame
    assert gradient.shape == (len(df),1)
    assert gradient.isna().iloc[0,0]
    assert gradient.iloc[1,0] == 0
    assert gradient.iloc[2,0] == 1/60

def test_gradient_dataframe_2col(constant_data_series):
    series = pd.Series([1, 1, 2, 1, 1], index=pd.date_range(start="2020", periods=5, freq="1min"))
    df = pd.concat([series.rename("col1"), series.rename("col2")*2], axis=1)

    detector = RangeDetector()
    gradient = detector._gradient(df)
    assert type(gradient) is pd.DataFrame
    assert gradient.shape == (len(df),2)
    assert gradient.isna().iloc[0,0]
    assert (gradient.iloc[1,:].values == np.array([0,0])).all()
    assert gradient.iloc[2,0] == 1/60
    assert gradient.iloc[2,1] == 2/60


def test_edge_cases():
    detector = RangeDetector()

    # Empty series
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        detector.detect(empty_series)

    # Empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        detector.detect(empty_df)

    # DataFrame with non-unique column names
    tmp= pd.Series([1, 2, 3, np.nan, 5], name="A")
    df_non_unique_cols = pd.concat([tmp, tmp * 2], axis=1)
    with pytest.raises(ValueError, match="DataFrame columns names must be unique."):
        detector.detect(df_non_unique_cols)

    # Series with non-numeric data
    #non_numeric_series = pd.Series(["a", "b", "c", "d"])
    #with pytest.raises(WrongInputDataTypeError, match="Input series must be numeric."):
    #    detector.detect(non_numeric_series)

    # All nan
    #all_nan_series = pd.Series([np.nan, np.nan, np.nan])
    #with pytest.raises(ValueError, match="Input data cannot be all NaN"):
    #    detector.detect(all_nan_series)