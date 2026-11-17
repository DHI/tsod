"""Tests for tsod.detectors.SeasonalProfileDetector."""

import numpy as np
import pandas as pd
import pytest

from tsod.custom_exceptions import InvalidArgumentError, NotIntegerError
from tsod.detectors import SeasonalProfileDetector

PERIOD = 24
N_CYCLES = 14


@pytest.fixture
def daily_cycle_with_noise():
    """Hourly flow over 14 days: a daily sine cycle (5 to 15) plus small seeded noise."""
    rng = np.random.default_rng(42)
    profile = 10 + 5 * np.sin(2 * np.pi * np.arange(PERIOD) / PERIOD)
    values = np.tile(profile, N_CYCLES) + rng.normal(0, 0.1, PERIOD * N_CYCLES)
    return pd.Series(values)


# --- Happy path ---


def test_flags_daytime_level_held_through_the_night(daily_cycle_with_noise):
    # The night minimum of day 7 (hours 15-21, values 5-7) stays at 12 instead.
    # Each value is normal for the series overall, but not at that time of day.
    night_hours = np.arange(7 * PERIOD + 15, 7 * PERIOD + 22)
    flow = daily_cycle_with_noise.copy()
    flow[night_hours] = 12.0

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(flow)

    assert anomalies[night_hours].all()


def test_perfectly_repeating_cycle_has_no_anomalies():
    profile = np.array([1.0, 3.0, 2.0, 5.0])
    flow = pd.Series(np.tile(profile, 10))

    anomalies = SeasonalProfileDetector(period=4).detect(flow)

    assert not anomalies.any()


@pytest.mark.parametrize(
    "direction, expect_high_flagged, expect_low_flagged",
    [
        ("both", True, True),
        ("positive", True, False),
        ("negative", False, True),
    ],
)
def test_direction_selects_which_deviations_are_flagged(
    daily_cycle_with_noise, direction, expect_high_flagged, expect_low_flagged
):
    high_index, low_index = 3 * PERIOD + 5, 9 * PERIOD + 5
    flow = daily_cycle_with_noise.copy()
    flow[high_index] += 3.0
    flow[low_index] -= 3.0

    anomalies = SeasonalProfileDetector(period=PERIOD, direction=direction).detect(flow)

    assert anomalies[high_index] == expect_high_flagged
    assert anomalies[low_index] == expect_low_flagged


def test_series_starting_mid_cycle_needs_no_alignment(daily_cycle_with_noise):
    flow = daily_cycle_with_noise.copy()
    flow[7 * PERIOD + 18] = 15.0  # daytime peak value at the night minimum
    flow_starting_at_hour_7 = flow.iloc[7:]

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(flow_starting_at_hour_7)

    assert anomalies[7 * PERIOD + 18]


def test_keeps_non_datetime_index(daily_cycle_with_noise):
    flow = daily_cycle_with_noise.copy()
    flow.index = np.arange(1000, 1000 + len(flow))

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(flow)

    pd.testing.assert_index_equal(anomalies.index, flow.index)


def test_dataframe_columns_are_profiled_independently(daily_cycle_with_noise):
    anomaly_index = 5 * PERIOD + 10
    shifted = daily_cycle_with_noise + 100.0
    shifted[anomaly_index] += 3.0
    df = pd.DataFrame({"normal": daily_cycle_with_noise, "with_anomaly": shifted})

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(df)

    assert anomalies.loc[anomaly_index, "with_anomaly"]
    assert not anomalies.loc[anomaly_index, "normal"]


# --- Edge cases ---


def test_missing_values_are_not_anomalies(daily_cycle_with_noise):
    missing_index, anomaly_index = 4 * PERIOD + 2, 8 * PERIOD + 2
    flow = daily_cycle_with_noise.copy()
    flow[missing_index] = np.nan
    flow[anomaly_index] += 3.0

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(flow)

    assert not anomalies[missing_index]
    assert anomalies[anomaly_index]


def test_fewer_than_four_cycles_are_not_evaluated(daily_cycle_with_noise):
    three_cycles = daily_cycle_with_noise.iloc[: 3 * PERIOD].copy()
    three_cycles[PERIOD + 5] = 1000.0

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(three_cycles)

    assert not anomalies.any()


def test_spread_is_shared_by_all_slots(daily_cycle_with_noise):
    # Hour 4 is almost noise-free, so a deviation of 0.2 is large for that hour
    # alone, but small compared with the noise level of the whole cycle (0.1).
    quiet_hour = np.arange(4, len(daily_cycle_with_noise), PERIOD)
    deviating_index = 6 * PERIOD + 4
    flow = daily_cycle_with_noise.copy()
    flow[quiet_hour] = 10.0
    flow[deviating_index] = 10.2

    anomalies = SeasonalProfileDetector(period=PERIOD).detect(flow)

    assert not anomalies[deviating_index]


def test_any_deviation_from_a_perfectly_repeating_cycle_is_an_anomaly():
    flow = pd.Series(np.tile([1.0, 2.0], 10))
    flow[6] = 1.5

    anomalies = SeasonalProfileDetector(period=2).detect(flow)

    assert anomalies[anomalies].index.tolist() == [6]


# --- Error cases ---


@pytest.mark.parametrize("period", [24.0, "24", True])
def test_rejects_non_integer_period(period):
    with pytest.raises(NotIntegerError):
        SeasonalProfileDetector(period=period)


@pytest.mark.parametrize("period", [1, 0, -24])
def test_rejects_period_shorter_than_two_steps(period):
    with pytest.raises(InvalidArgumentError):
        SeasonalProfileDetector(period=period)


@pytest.mark.parametrize("threshold", [0, -3.0])
def test_rejects_non_positive_threshold(threshold):
    with pytest.raises(InvalidArgumentError):
        SeasonalProfileDetector(period=PERIOD, threshold=threshold)


def test_rejects_unknown_direction():
    with pytest.raises(ValueError, match="not a valid direction"):
        SeasonalProfileDetector(period=PERIOD, direction="up")
