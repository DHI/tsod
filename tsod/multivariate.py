import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ScaledReferenceDetector:
    def __init__(self, *, rtol=0.00001, atol=1e-8, factor=1.0, offset=0.0):

        self.rtol = rtol
        self.atol = atol
        self.factor = factor
        self.offset = offset

    def fit(self, value: pd.Series, reference: pd.Series) -> None:
        """Fit scaling coefficients"""

        lr = LinearRegression()
        lr.fit(reference.values.reshape(-1, 1), value.values)
        self.factor = lr.coef_[0]
        self.offset = lr.intercept_

        # TODO fit rtol and/ or atol as well

    def detect(self, value: pd.Series, reference: pd.Series) -> pd.Series:

        transformed = self._transform(reference)

        res = pd.Series(~np.isclose(transformed, value, rtol=self.rtol, atol=self.atol))

        return res

    def _transform(self, reference: pd.Series) -> pd.Series:
        return reference * self.factor + self.offset
