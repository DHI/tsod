import os
import numpy as np
import pandas as pd
import tsod
from tsod import (
    RangeDetector,
    ConstantValueDetector,
    CombinedDetector,
    DriftDetector,
    CusumDriftDetector,
)


def test_save_and_load(tmp_path):
    combined = CombinedDetector(
        [
            ConstantValueDetector(),
            RangeDetector(max_value=2.0),
        ]
    )

    path = tmp_path / "combined.joblib"
    combined.save(path)

    loaded = tsod.load(path)

    assert isinstance(loaded, CombinedDetector)


def test_load():
    path_to_tests_super_folder = os.path.abspath(__file__).split("tests")[0]
    filename = os.path.join(
        path_to_tests_super_folder, "tests", "data", "combined.joblib"
    )

    loaded = tsod.load(filename)

    assert isinstance(loaded, CombinedDetector)


def test_save_and_load_filename(tmpdir):
    combined = CombinedDetector(
        [
            ConstantValueDetector(),
            RangeDetector(max_value=2.0),
        ]
    )

    filename = os.path.join(tmpdir, "combined.joblib")
    combined.save(filename)

    loaded = tsod.load(filename)

    assert isinstance(loaded, CombinedDetector)


def test_save_and_load_drift_detectors(tmp_path):
    """A fitted drift detector must keep detecting the same after a round trip."""
    time = pd.date_range(start="2020", periods=200, freq="1h")
    rng = np.random.default_rng(42)
    normal = pd.Series(rng.normal(size=200), index=time)
    drifting = normal + np.linspace(0.0, 20.0, 200)

    combined = CombinedDetector(
        [
            DriftDetector(window_size=24),
            CusumDriftDetector(threshold=15.0),
        ]
    ).fit(normal)

    path = tmp_path / "drift.joblib"
    combined.save(path)
    loaded = tsod.load(path)

    assert isinstance(loaded, CombinedDetector)
    assert isinstance(loaded[0], DriftDetector)
    assert isinstance(loaded[1], CusumDriftDetector)
    assert (loaded.detect(drifting) == combined.detect(drifting)).all()
