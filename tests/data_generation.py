import numpy as np


def create_random_walk_with_outliers(
    n_steps, t0=0, outlier_fraction=0.1, outlier_scale=10, seed=42
):
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
    rng = np.random.default_rng(seed)
    possible_steps = [-1, 1]
    random_steps = rng.choice(a=possible_steps, size=n_steps)
    random_walk = np.append(t0, random_steps[:-1]).cumsum(axis=0)

    # Add outliers
    random_walk_with_outliers = random_walk.copy()
    outlier_indices = rng.integers(0, n_steps, n_outliers)
    random_walk_with_outliers[outlier_indices] += (
        random_steps[outlier_indices] * outlier_scale
    )

    return random_walk_with_outliers, sorted(outlier_indices), random_walk


def create_drifting_series(
    n_steps, drift_start, drift_per_step, noise_scale=1.0, seed=42
):
    """Generate a noisy stationary series that starts drifting part way through.

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
    seed : int
        Random seed

    Returns
    -------
    drifting : np.ndarray
        The generated time series, drifting from `drift_start` onwards.
    normal : np.ndarray
        The same series without the drift, i.e. noise only.
    """
    assert 0 <= drift_start <= n_steps

    rng = np.random.default_rng(seed)
    normal = rng.normal(scale=noise_scale, size=n_steps)

    drift = np.zeros(n_steps)
    n_drifting = n_steps - drift_start
    drift[drift_start:] = np.arange(n_drifting) * drift_per_step

    return normal + drift, normal
