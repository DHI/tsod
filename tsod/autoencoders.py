import pandas as pd
from pyod.models.auto_encoder import AutoEncoder as AutoEncoderPyod

from tsod.detectors import Detector


class AutoEncoder(Detector):
    def __init__(self, **kwargs):
        super().__init__()
        self._model = AutoEncoderPyod(**kwargs)

    def _fit(self, data):

        self._model.fit(data)

        return self

    def _detect(self, data):
        return self._model.predict(data)

    def validate(self, data):
        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        else:
            return data

    def __str__(self):
        return f"{self.__class__.__name__}({self._model})"