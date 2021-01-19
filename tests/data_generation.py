import numpy as np


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
    random_walk_with_outliers = random_walk.copy()
    outlier_indices = np.random.randint(0, n_steps, n_outliers)
    random_walk_with_outliers[outlier_indices] += random_steps[outlier_indices] * outlier_scale

    return random_walk_with_outliers, sorted(outlier_indices), random_walk
