import os
import tsod
from tsod import RangeDetector, ConstantValueDetector, CombinedDetector


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
