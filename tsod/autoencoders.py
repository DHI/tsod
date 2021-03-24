import pandas as pd
from pyod.models.auto_encoder import AutoEncoder as AutoEncoderPyod

from tsod.detectors import BaseDetector


class AutoEncoder(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__()
        self._model = AutoEncoderPyod(**kwargs)

    def fit(self, data):
        data = self._validate(data)
        self._model.fit(data)

        return self

    def detect(self, data):
        data = self._validate(data)
        return self._model.predict(data)

    def _validate(self, data):
        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        else:
            return data

    def __str__(self):
        return f"{self.__class__.__name__}({self._model})"