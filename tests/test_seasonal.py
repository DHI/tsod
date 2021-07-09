import pytest
import pandas as pd

from tsod.detectors import SeasonalDetector


@pytest.fixture
def raan():
    return pd.read_csv("tests/data/raan.csv", parse_dates=True, index_col=0)["flow"]


def test_seasonal(raan):

    detector = SeasonalDetector()
    detector.fit(raan)

    anom = detector.detect(raan)

    assert len(anom) == len(raan)
