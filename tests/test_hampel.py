import numpy as np
import pandas as pd
import pytest

from anomalydetection.detectors import HampelDetector


@pytest.fixture
def data_series():
    n_steps = 100
    time_series_with_outliers, outlier_indices = create_random_walk_with_outliers(n_steps)
    time = pd.date_range(start='2020', periods=n_steps, freq='1H')
    return pd.Series(time_series_with_outliers, index=time), outlier_indices


def test_hampel_detector(data_series):
    data, expected_anomalies_indices = data_series
    detector = HampelDetector()
    anomalies = detector.detect(data)
    anomalies_indices = np.array(np.where(anomalies)).flatten()
    # Validate if the found anomalies are also in the expected anomaly set
    assert all(i in expected_anomalies_indices for i in anomalies_indices)


def create_random_walk_with_outliers(n_steps, t0=0, outlier_fraction=0.1, outlier_scale=10, seed=42):
    """
    Generate a random walk time series with random outlier peaks.

    Parameters
    ------------
    n_steps : int
        Length of the time series to be generated.
    t0 : int
        Time series initial value.
    outlier_fraction : float
        Fraction of outliers to be generated in series [0-1].
    outlier_scale : float
        Scalar by which to multiply the RW increment to create an outlier.
    seed : int
        Random seed

    Returns
    -------
    random_walk : np.ndarray
        The generated random walk time series with outliers.
    outlier_indices : np.ndarray
        The indices of the introduced outliers.
    """
    assert 0 <= outlier_fraction <= 1
    n_outliers = int(outlier_fraction * n_steps)

    # Simulate random walk
    np.random.seed(seed)
    possible_steps = [-1, 1]
    random_steps = np.random.choice(a=possible_steps, size=n_steps)
    random_walk = np.append(t0, random_steps[:-1]).cumsum(axis=0)

    # Add outliers
    outlier_indices = np.random.randint(0, n_steps, n_outliers)
    random_walk[outlier_indices] += random_steps[outlier_indices] * outlier_scale

    return random_walk, sorted(outlier_indices)
