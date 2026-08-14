import pytest
import numpy as np
import pandas as pd

from tsod.custom_exceptions import InvalidArgumentError, NotFittedError
from tsod.detectors import CombinedDetector, DriftDetector, _rolling_slope
from tsod.cusum import CusumDriftDetector

from tests.data_generation import create_drifting_series


@pytest.fixture
def drift_data_series():
    n_steps = 400
    drift_start = 200
    drifting, normal = create_drifting_series(
        n_steps, drift_start=drift_start, drift_per_step=0.2, noise_scale=1.0
    )
    time = pd.date_range(start="2020", periods=n_steps, freq="1h")
    return (
        pd.Series(normal, index=time),
        pd.Series(drifting, index=time),
        drift_start,
    )


def test_rolling_slope_recovers_known_trend():
    time = pd.date_range(start="2020", periods=50, freq="1h")
    # exactly 1 unit per hour
    data = pd.Series(np.arange(50, dtype=float), index=time).to_frame()

    slope = _rolling_slope(data, window_size=10).iloc[:, 0]

    assert slope.iloc[:9].isna().all()  # incomplete windows
    assert slope.iloc[9:].to_numpy() == pytest.approx(1.0 / 3600.0)


def test_rolling_slope_ignores_nan():
    time = pd.date_range(start="2020", periods=50, freq="1h")
    values = np.arange(50, dtype=float)
    values[[3, 17, 42]] = np.nan
    data = pd.Series(values, index=time).to_frame()

    slope = _rolling_slope(data, window_size=10).iloc[:, 0]

    # A gap reduces the number of points in a window, but not the trend
    assert slope.iloc[9:].to_numpy() == pytest.approx(1.0 / 3600.0)


def test_rolling_slope_non_uniform_dt():
    ind = pd.DatetimeIndex(
        [
            "2020-01-01 01:00:00",
            "2020-01-01 01:00:30",
            "2020-01-01 01:02:00",
            "2020-01-01 01:04:00",
            "2020-01-01 01:08:00",
        ]
    )
    # value equals elapsed seconds, so the trend is exactly 1 per second
    data = pd.Series(index=ind, data=[0.0, 30.0, 120.0, 240.0, 480.0]).to_frame()

    slope = _rolling_slope(data, window_size=3).iloc[:, 0]

    assert slope.iloc[2:].to_numpy() == pytest.approx(1.0)


def test_rolling_slope_long_series_precision():
    """Sums of squares must not lose the trend on a multi-year series."""
    time = pd.date_range(start="2000", periods=20000, freq="1h")
    data = pd.Series(np.arange(20000, dtype=float), index=time).to_frame()

    slope = _rolling_slope(data, window_size=100).iloc[:, 0]

    assert slope.iloc[99:].to_numpy() == pytest.approx(1.0 / 3600.0, rel=1e-9)


def test_drift_detector(drift_data_series):
    normal, drifting, drift_start = drift_data_series

    detector = DriftDetector(window_size=48)
    detector.fit(normal)
    anomalies = detector.detect(drifting)

    assert isinstance(anomalies, pd.Series)
    assert len(anomalies) == len(drifting)

    # Nothing is flagged before the drift starts, and it is caught once a full
    # window lies inside the drifting part
    assert not anomalies.iloc[:drift_start].any()
    assert anomalies.iloc[drift_start + 48 :].all()


def test_drift_detector_ignores_noise(drift_data_series):
    normal, _, _ = drift_data_series

    detector = DriftDetector(window_size=48)
    detector.fit(normal)

    # Fitting on the normal data means its own steepest trend is not an anomaly
    assert not detector.detect(normal).any()


def test_drift_detector_explicit_rate():
    time = pd.date_range(start="2020", periods=200, freq="1h")
    # 0.5 per day
    data = pd.Series(np.arange(200) * 0.5 / 24.0, index=time)

    too_strict = DriftDetector(window_size=24, max_drift_rate=0.1 / 86400.0)
    assert too_strict.detect(data).iloc[23:].all()

    tolerant = DriftDetector(window_size=24, max_drift_rate=1.0 / 86400.0)
    assert not tolerant.detect(data).any()


def test_drift_detector_recovers_after_step_change():
    """A single jump is not drift, GradientDetector is the tool for that."""
    time = pd.date_range(start="2020", periods=200, freq="1h")
    values = np.zeros(200)
    values[100:] = 10.0
    data = pd.Series(values, index=time)

    detector = DriftDetector(window_size=48, max_drift_rate=1.0 / 86400.0)
    anomalies = detector.detect(data)

    # The step looks like drift while it sits inside the window, but not once the
    # window has moved past it
    assert not anomalies.iloc[:100].any()
    assert not anomalies.iloc[160:].any()


def test_drift_detector_direction():
    time = pd.date_range(start="2020", periods=100, freq="1h")
    rising = pd.Series(np.arange(100) * 0.1, index=time)
    falling = pd.Series(-np.arange(100) * 0.1, index=time)

    max_rate = 0.01 / 86400.0

    positive = DriftDetector(
        window_size=24, max_drift_rate=max_rate, direction="positive"
    )
    assert positive.detect(rising).iloc[23:].all()
    assert not positive.detect(falling).any()

    negative = DriftDetector(
        window_size=24, max_drift_rate=max_rate, direction="negative"
    )
    assert negative.detect(falling).iloc[23:].all()
    assert not negative.detect(rising).any()


def test_drift_detector_centered():
    time = pd.date_range(start="2020", periods=100, freq="1h")
    data = pd.Series(np.arange(100) * 0.1, index=time)

    detector = DriftDetector(window_size=25, max_drift_rate=0.0, center=True)
    anomalies = detector.detect(data)

    # Centered labels leave half a window undetermined at each end
    assert not anomalies.iloc[:12].any()
    assert not anomalies.iloc[-12:].any()
    assert anomalies.iloc[12:-12].all()


def test_drift_detector_multicol(drift_data_series):
    normal, drifting, drift_start = drift_data_series

    df = pd.concat([normal.rename("a"), drifting.rename("b")], axis=1)

    detector = DriftDetector(window_size=48, max_drift_rate=0.05 / 3600.0)
    anomalies = detector.detect(df)

    assert isinstance(anomalies, pd.DataFrame)
    assert anomalies.shape == df.shape
    assert not anomalies["a"].any()
    assert anomalies["b"].iloc[drift_start + 48 :].all()


def test_drift_detector_requires_datetime_index():
    detector = DriftDetector(window_size=5)
    data = pd.Series(np.arange(20, dtype=float))

    with pytest.raises(
        ValueError,
        match="Slope calculation requires a DatetimeIndex. Got RangeIndex instead",
    ):
        detector.detect(data)


def test_drift_detector_invalid_arguments():
    with pytest.raises(ValueError, match="window_size must be at least 2"):
        DriftDetector(window_size=1)

    with pytest.raises(ValueError, match="max_drift_rate must be non-negative"):
        DriftDetector(max_drift_rate=-1.0)

    with pytest.raises(ValueError, match="is not a valid direction"):
        DriftDetector(direction="sideways")


def test_drift_detector_str():
    detector = DriftDetector(window_size=10, max_drift_rate=1.0 / 86400.0)

    assert "DriftDetector" in str(detector)
    assert "1.0/day" in str(detector)


def test_cusum_detector(drift_data_series):
    normal, drifting, drift_start = drift_data_series

    detector = CusumDriftDetector()
    detector.fit(normal)
    anomalies = detector.detect(drifting)

    assert isinstance(anomalies, pd.Series)
    assert len(anomalies) == len(drifting)
    assert not anomalies.iloc[:drift_start].any()
    assert anomalies.iloc[-1]


def test_cusum_detects_drift_earlier_than_trend(drift_data_series):
    """The point of accumulating evidence instead of windowing it."""
    normal, drifting, _ = drift_data_series

    cusum = CusumDriftDetector().fit(normal)
    trend = DriftDetector(window_size=48).fit(normal)

    first_cusum = cusum.detect(drifting).to_numpy().argmax()
    first_trend = trend.detect(drifting).to_numpy().argmax()

    assert first_cusum < first_trend


def test_cusum_detector_ignores_noise(drift_data_series):
    """Noise alone must not accumulate into a sustained alarm.

    Unlike the other detectors, a control chart has a false alarm rate by design,
    set by slack and threshold, so a few isolated flags are expected rather than
    none at all.
    """
    normal, _, _ = drift_data_series

    detector = CusumDriftDetector().fit(normal)
    anomalies = detector.detect(normal)

    assert anomalies.mean() < 0.05


def test_cusum_detector_threshold_controls_false_alarms(drift_data_series):
    normal, _, _ = drift_data_series

    sensitive = CusumDriftDetector(threshold=2.0).fit(normal)
    conservative = CusumDriftDetector(threshold=15.0).fit(normal)

    assert sensitive.detect(normal).sum() > conservative.detect(normal).sum()
    assert not conservative.detect(normal).any()


def test_cusum_detector_flags_whole_drifted_period():
    time = pd.date_range(start="2020", periods=200, freq="1h")
    # a sustained offset, not a trend
    shifted = pd.Series(np.concatenate([np.zeros(100), np.ones(100)]), index=time)

    detector = CusumDriftDetector(slack=0.1, threshold=2.0)
    # a constant normal level has no scale to fit, so set it directly
    detector._center = 0.0
    detector._scale = 1.0

    anomalies = detector.detect(shifted)

    assert not anomalies.iloc[:100].any()
    # The sums are not reset on alarm, so the flag stays up for the whole period
    assert anomalies.iloc[110:].all()


def test_cusum_detector_direction():
    time = pd.date_range(start="2020", periods=200, freq="1h")
    up = pd.Series(np.concatenate([np.zeros(100), np.ones(100)]), index=time)
    down = -up

    cases = [("positive", True, False), ("negative", False, True), ("both", True, True)]
    for direction, expected_up, expected_down in cases:
        detector = CusumDriftDetector(slack=0.1, threshold=2.0, direction=direction)
        detector._center = 0.0
        detector._scale = 1.0

        assert detector.detect(up).any() == expected_up
        assert detector.detect(down).any() == expected_down


def test_cusum_detector_handles_nan():
    time = pd.date_range(start="2020", periods=200, freq="1h")
    values = np.concatenate([np.zeros(100), np.ones(100)])
    values[[5, 120, 150]] = np.nan
    data = pd.Series(values, index=time)

    detector = CusumDriftDetector(slack=0.1, threshold=2.0)
    detector._center = 0.0
    detector._scale = 1.0

    anomalies = detector.detect(data)

    assert not anomalies.iloc[:100].any()
    assert anomalies.iloc[-1]


def test_cusum_detector_multicol(drift_data_series):
    normal, drifting, _ = drift_data_series

    df = pd.concat([normal.rename("a"), drifting.rename("b")], axis=1)

    detector = CusumDriftDetector(threshold=15.0).fit(normal)
    anomalies = detector.detect(df)

    assert isinstance(anomalies, pd.DataFrame)
    assert anomalies.shape == df.shape
    assert not anomalies["a"].any()
    assert anomalies["b"].iloc[-1]


def test_cusum_detector_requires_fit(drift_data_series):
    _, drifting, _ = drift_data_series

    detector = CusumDriftDetector()
    with pytest.raises(NotFittedError):
        detector.detect(drifting)


def test_cusum_detector_fit_on_constant_data():
    data = pd.Series(np.ones(100))

    detector = CusumDriftDetector()
    with pytest.raises(ValueError, match="Could not determine a scale"):
        detector.fit(data)


def test_cusum_detector_robust_scale():
    """Outliers in the training data must not inflate the scale."""
    rng = np.random.default_rng(42)
    values = rng.normal(size=200)
    clean = pd.Series(values.copy())
    values[[10, 50, 120]] = 100.0
    contaminated = pd.Series(values)

    scale_clean = CusumDriftDetector().fit(clean)._scale
    scale_contaminated = CusumDriftDetector().fit(contaminated)._scale

    assert scale_contaminated == pytest.approx(scale_clean, rel=0.1)


def test_cusum_detector_invalid_arguments():
    with pytest.raises(InvalidArgumentError, match="slack must be non-negative"):
        CusumDriftDetector(slack=-1.0)

    with pytest.raises(InvalidArgumentError, match="threshold must be positive"):
        CusumDriftDetector(threshold=0.0)

    with pytest.raises(ValueError, match="is not a valid direction"):
        CusumDriftDetector(direction="sideways")


def test_cusum_detector_str():
    detector = CusumDriftDetector(slack=0.5, threshold=5.0)

    assert "CusumDriftDetector" in str(detector)


@pytest.fixture
def tidal_data_series():
    """A semi-diurnal tide, with the sensor fouling from day 20 onwards."""
    dt_minutes, days, drift_per_day = 10, 40, 0.05
    n_steps = days * 24 * (60 // dt_minutes)
    time = pd.date_range(start="2020", periods=n_steps, freq=f"{dt_minutes}min")

    hours = np.arange(n_steps) * dt_minutes / 60.0
    rng = np.random.default_rng(1)
    tide = np.sin(2 * np.pi * hours / 12.42) + 0.02 * rng.normal(size=n_steps)
    drift = np.where(hours / 24 > 20, (hours / 24 - 20) * drift_per_day, 0.0)

    points_per_cycle = int(round(12.42 * 60 / dt_minutes))
    return (
        pd.Series(tide, index=time),
        pd.Series(tide + drift, index=time),
        points_per_cycle,
        hours / 24,
    )


def test_drift_detector_window_must_span_whole_cycles(tidal_data_series):
    """A window shorter than the period measures the tide, not the drift."""
    clean, fouled, points_per_cycle, day = tidal_data_series

    half_cycle = DriftDetector(window_size=points_per_cycle // 2).fit(clean)
    many_cycles = DriftDetector(window_size=10 * points_per_cycle).fit(clean)

    # The fitted rate collapses once the tide averages out within the window
    assert half_cycle.max_drift_rate > 1.0 / 86400.0
    assert many_cycles.max_drift_rate < 0.05 / 86400.0

    drifted = fouled.index[day > 21]
    assert half_cycle.detect(fouled)[drifted].mean() < 0.05
    assert many_cycles.detect(fouled)[drifted].mean() > 0.5

    # Neither raises a false alarm on the undrifted part
    before = fouled.index[day <= 20]
    assert not half_cycle.detect(fouled)[before].any()
    assert not many_cycles.detect(fouled)[before].any()


@pytest.mark.parametrize(
    "detector",
    [
        DriftDetector(window_size=5, max_drift_rate=0.0),
        CusumDriftDetector(slack=0.1, threshold=2.0),
    ],
    ids=["drift", "cusum"],
)
def test_drift_detectors_edge_cases(detector):
    if isinstance(detector, CusumDriftDetector):
        detector._center = 0.0
        detector._scale = 1.0

    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    assert len(detector.detect(empty)) == 0

    # Too short to determine a trend, but must not raise
    for n in (1, 2, 3):
        short = pd.Series(
            np.arange(n, dtype=float), index=pd.date_range("2020", periods=n, freq="1h")
        )
        assert len(detector.detect(short)) == n

    all_nan = pd.Series(
        np.full(20, np.nan), index=pd.date_range("2020", periods=20, freq="1h")
    )
    assert not detector.detect(all_nan).any()


def test_drift_detector_rejects_duplicate_timestamps():
    data = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"]),
    )

    with pytest.raises(ValueError, match="Index must be monotonically increasing"):
        DriftDetector(window_size=2).detect(data)


def test_cusum_detector_fit_on_all_nan_data():
    data = pd.Series(np.full(20, np.nan))

    with pytest.raises(ValueError, match="Input data contains no valid values"):
        CusumDriftDetector().fit(data)


def test_drift_detectors_combine(drift_data_series):
    normal, drifting, drift_start = drift_data_series

    combined = CombinedDetector(
        [DriftDetector(window_size=48), CusumDriftDetector(threshold=15.0)]
    )
    combined.fit(normal)
    anomalies = combined.detect(drifting)

    assert not anomalies.iloc[:drift_start].any()
    assert anomalies.iloc[-1]
