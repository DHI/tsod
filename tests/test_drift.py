import pytest
import numpy as np
import pandas as pd

from tsod.detectors import (
    CombinedDetector,
    DriftDetector,
    RangeDetector,
)


def create_spiky_drifting_series(
    n_steps,
    drift_start,
    drift_per_step,
    noise_scale=1.0,
    n_spikes=20,
    spike_scale=30.0,
    max_spike_length=40,
    seed=42,
):
    """Generate a drifting series contaminated by large one-sided event spikes.

    Models an event-driven signal, e.g. a river level with storm peaks far above
    its baseline, on which a slow drift has to be found. The spikes are one-sided
    and decay, so they behave like real events rather than symmetric outliers.

    Parameters
    ------------
    n_steps : int
        Length of the time series to be generated.
    drift_start : int
        Index at which the drift begins.
    drift_per_step : float
        Amount added to the signal per step once the drift has begun. A negative
        value drifts downwards.
    noise_scale : float
        Standard deviation of the noise added to the signal.
    n_spikes : int
        Number of event spikes to add.
    spike_scale : float
        Mean height of an event spike, in the units of the data.
    max_spike_length : int
        Longest an event may last, in steps. Lengths are drawn up to this.
    seed : int
        Random seed

    Returns
    -------
    drifting : np.ndarray
        The generated time series, with spikes throughout and drifting from
        `drift_start` onwards.
    normal : np.ndarray
        The same series without the drift, i.e. noise and spikes only.
    """
    assert 0 <= drift_start <= n_steps

    rng = np.random.default_rng(seed)
    normal = rng.normal(scale=noise_scale, size=n_steps)

    for position in rng.choice(n_steps, n_spikes, replace=False):
        length = min(rng.integers(2, max_spike_length + 1), n_steps - position)
        decay = np.exp(-np.arange(length) / max(length / 3.0, 1.0))
        normal[position : position + length] += rng.exponential(spike_scale) * decay

    drift = np.zeros(n_steps)
    drift[drift_start:] = np.arange(n_steps - drift_start) * drift_per_step

    return normal + drift, normal


@pytest.fixture
def spiky_drift_series():
    """A drifting signal buried under events far larger than the drift."""
    n_steps = 4000
    drift_start = 2000
    drifting, normal = create_spiky_drifting_series(
        n_steps,
        drift_start=drift_start,
        drift_per_step=0.02,
        noise_scale=1.0,
        n_spikes=40,
        spike_scale=40.0,
    )
    time = pd.date_range(start="2020", periods=n_steps, freq="1h")
    return (
        pd.Series(normal, index=time),
        pd.Series(drifting, index=time),
        drift_start,
    )


def _drift(values, window, lookback, index=None):
    """The drift, from the detector's own calculation."""
    data = pd.Series(values, dtype=float, index=index)
    detector = DriftDetector(window=window, lookback=lookback)
    return detector._drift(data.to_frame()).iloc[:, 0]


def test_drift_is_zero_on_a_flat_signal():
    drift = _drift(np.full(300, 7.0), window=10, lookback=50)

    assert drift.iloc[: 10 + 50 - 1].isna().all()  # no full window and lookback yet
    assert (drift.iloc[10 + 50 - 1 :] == 0.0).all()


def test_drift_is_undamped():
    """The whole point: the drift is the distance travelled over the lookback.

    Non-overlapping windows report the full `rate * lookback`. Nesting the two
    windows instead, so that the earlier one contains the later one, would
    attenuate this to `rate * (lookback - window) / 2`.
    """
    rate, window, lookback = 0.3, 10, 50
    values = np.arange(400) * rate

    drift = _drift(values, window=window, lookback=lookback)

    assert drift.iloc[window + lookback :].to_numpy() == pytest.approx(rate * lookback)
    # Comfortably above what a nested pair of windows could have reported
    assert drift.iloc[-1] > rate * (lookback - window) / 2

    # And it scales with the lookback, since that is where the signal comes from
    doubled = _drift(values, window=window, lookback=2 * lookback)
    assert doubled.iloc[-1] == pytest.approx(2 * drift.iloc[-1])


def test_drift_ignores_a_spike():
    values = np.zeros(400)
    values[200:203] = 50.0

    drift = _drift(values, window=10, lookback=50)

    assert np.nanmax(np.abs(drift.to_numpy())) == 0.0


def test_drift_reports_a_step_for_one_lookback():
    window, lookback, height = 10, 50, 4.0
    values = np.zeros(400)
    values[200:] = height

    drift = _drift(values, window=window, lookback=lookback)

    # Full height while the step is behind the baseline but not yet behind the
    # reference, i.e. for about one lookback, and then nothing
    assert np.isclose(drift, height).sum() == pytest.approx(lookback, abs=window)
    assert drift.iloc[-1] == 0.0


def test_drift_counts_points_not_time():
    """A count of points reaches further back once samples go missing.

    This is the price of taking only ints: `lookback` is a number of samples, so
    with gaps it spans more time than intended and overstates the drift.
    """
    rate = 0.3
    time = pd.date_range(start="2020", periods=400, freq="1h")
    full = pd.Series(np.arange(400) * rate, index=time)
    rng = np.random.default_rng(0)
    sparse = full[rng.random(400) > 0.4]

    evenly_spaced = _drift(full.to_numpy(), window=10, lookback=50, index=time)
    with_gaps = _drift(sparse.to_numpy(), window=10, lookback=50, index=sparse.index)

    assert evenly_spaced.iloc[-1] == pytest.approx(rate * 50)
    assert with_gaps.iloc[-1] > 1.5 * rate * 50


def test_drift_rejects_durations():
    """Only whole numbers of points, so a duration is refused outright."""
    for bad in ("30D", pd.Timedelta("30D"), 10.0, True):
        with pytest.raises(ValueError, match="must be a number of points"):
            DriftDetector(window=bad, lookback=1000)
        with pytest.raises(ValueError, match="must be a number of points"):
            DriftDetector(window=10, lookback=bad)


def test_drift_detects_drift_buried_under_events(spiky_drift_series):
    """Events forty times the noise must not stop the drift being found."""
    normal, drifting, drift_start = spiky_drift_series

    detector = DriftDetector(window=200, lookback=800).fit(normal)
    anomalies = detector.detect(drifting)

    assert not detector.detect(normal).any()
    assert not anomalies.iloc[:drift_start].any()
    assert anomalies.iloc[-1]

    # The criterion learned from the spiky clean data is set by the drift-free
    # baseline wander, not by the events, so it stays far below the drift the
    # series eventually reaches
    assert detector.drift_limit < 0.02 * 800


def test_drift_fit_matches_an_explicit_shift(spiky_drift_series):
    normal, drifting, _ = spiky_drift_series

    fitted = DriftDetector(window=200, lookback=800).fit(normal)
    explicit = DriftDetector(window=200, lookback=800, drift_limit=fitted.drift_limit)

    assert (fitted.detect(drifting) == explicit.detect(drifting)).all()


def test_drift_without_criterion_flags_nothing():
    """An unset limit is infinite, as for RangeDetector, so nothing exceeds it."""
    data = pd.Series(np.arange(300) * 0.1)

    assert not DriftDetector(window=10, lookback=50).detect(data).any()


def test_drift_direction():
    rising = pd.Series(np.arange(400) * 0.3)
    falling = pd.Series(-np.arange(400) * 0.3)

    positive = DriftDetector(
        window=10, lookback=50, drift_limit=1.0, direction="positive"
    )
    negative = DriftDetector(
        window=10, lookback=50, drift_limit=1.0, direction="negative"
    )

    assert positive.detect(rising).any()
    assert not positive.detect(falling).any()
    assert negative.detect(falling).any()
    assert not negative.detect(rising).any()


def test_drift_counts_need_no_datetime_index():
    data = pd.Series(np.arange(400) * 0.3)  # RangeIndex

    assert DriftDetector(window=10, lookback=50, drift_limit=1.0).detect(data).any()


def test_drift_multicol(spiky_drift_series):
    normal, drifting, drift_start = spiky_drift_series
    normal_df = pd.DataFrame({"a": normal, "b": normal})
    drifting_df = pd.DataFrame({"a": normal, "b": drifting})

    detector = DriftDetector(window=200, lookback=800).fit(normal)
    anomalies = detector.detect(drifting_df)

    assert isinstance(anomalies, pd.DataFrame)
    assert not detector.detect(normal_df).any().any()
    assert not anomalies["a"].any()
    assert anomalies["b"].iloc[-1]


def test_drift_invalid_arguments():
    with pytest.raises(ValueError, match="window must be at least 1"):
        DriftDetector(window=0, lookback=50)

    with pytest.raises(ValueError, match="lookback must be at least window"):
        DriftDetector(window=50, lookback=10)

    with pytest.raises(ValueError, match="drift_limit must be non-negative"):
        DriftDetector(window=10, lookback=50, drift_limit=-1.0)

    with pytest.raises(ValueError, match="not a valid direction"):
        DriftDetector(window=10, lookback=50, direction="sideways")


def test_drift_edge_cases():
    detector = DriftDetector(window=10, lookback=50, drift_limit=1.0)

    assert detector.detect(pd.Series([], dtype=float)).empty

    # Shorter than window + lookback, so nothing can be said and nothing is flagged
    assert not detector.detect(pd.Series(np.arange(20) * 5.0)).any()

    all_nan = pd.Series(np.full(300, np.nan))
    assert not detector.detect(all_nan).any()

    # A criterion cannot be learned from data that yields no drift at all
    assert DriftDetector(window=10, lookback=50).fit(all_nan).drift_limit == 0.0


def test_drift_str():
    detector = DriftDetector(window=144, lookback=1008, drift_limit=0.25)

    assert "DriftDetector" in str(detector)
    assert "144" in str(detector)
    assert "1008" in str(detector)
    assert "0.25" in str(detector)
    assert "not set" in str(DriftDetector(window=10, lookback=50))


def test_drift_combines(spiky_drift_series):
    normal, drifting, drift_start = spiky_drift_series

    combined = CombinedDetector(
        [DriftDetector(window=200, lookback=800), RangeDetector()]
    )
    combined.fit(normal)
    anomalies = combined.detect(drifting)

    assert not anomalies.iloc[:drift_start].any()
    assert anomalies.iloc[-1]
