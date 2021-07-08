import pandas as pd
import pytest

from tsod.multivariate import ScaledReferenceDetector


@pytest.fixture
def discharge():
    df = pd.read_csv(
        "tests/data/discharge.csv", parse_dates=True, index_col=0, comment="#"
    )

    return df


def test_reference(discharge):

    detector = ScaledReferenceDetector(rtol=0.1)
    anomalies = detector.detect(value=discharge["raan"], reference=discharge["vege"])

    assert len(anomalies) == len(discharge)
    assert sum(anomalies) > 0


def test_reference_fit(discharge):

    detector = ScaledReferenceDetector(rtol=0.5, atol=0.5)
    detector.fit(value=discharge["raan"], reference=discharge["vege"])

    assert detector.factor != 1.0

    anomalies = detector.detect(value=discharge["raan"], reference=discharge["vege"])
    assert sum(anomalies) == 10  # TODO
