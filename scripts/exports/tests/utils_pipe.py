import pickle

import numpy as np
from tensorflow import keras

MUONLL_ERRORCODE = -1000.0
PROBMU_ERRORCODE = -1.0


class GanPipe:
    def __init__(self, filepath):
        with open(f"{filepath}/tX.pkl", "rb") as file:
            self._tX = pickle.load(file)
        with open(f"{filepath}/tY.pkl", "rb") as file:
            self._tY = pickle.load(file)
        self._model = keras.models.load_model(f"{filepath}/saved_generator")

    def predict(self, x, rnd_noise, batch_size=None):
        prep_x = self._tX.transform(x)
        out = self._model.predict(
            np.concatenate([prep_x, rnd_noise], axis=1), batch_size=batch_size
        )
        post_out = self._tY.inverse_transform(out)
        return post_out


class isMuonPipe:
    def __init__(self, filepath):
        with open(f"{filepath}/tX.pkl", "rb") as file:
            self._tX = pickle.load(file)
        self._model = keras.models.load_model(f"{filepath}/saved_model")

    def predict_proba(self, x, batch_size=None):
        prep_x = self._tX.transform(x)
        out = self._model.predict(prep_x, batch_size=batch_size)
        return out


class FullPipe:
    def __init__(self, ml_pipes):
        self._rich = ml_pipes["Rich"]
        self._muon = ml_pipes["Muon"]
        self._gpidmu = ml_pipes["GlobalPIDmu"]
        self._gpidh = ml_pipes["GlobalPIDh"]

    def predict(self, x, rnd_noise, ismuon_flag, batch_size=None):
        rich_out = self._rich.predict(
            x, rnd_noise=rnd_noise[:, 0o000:0o100], batch_size=batch_size
        )
        muon_out = self._muon.predict(
            x, rnd_noise=rnd_noise[:, 0o100:0o200], batch_size=batch_size
        )
        gpidmu_out = self._gpidmu.predict(
            np.concatenate([x, rich_out, muon_out], axis=1),
            rnd_noise=rnd_noise[:, 0o200:0o300],
            batch_size=batch_size,
        )
        gpidh_out = self._gpidh.predict(
            np.concatenate([x, rich_out], axis=1),
            rnd_noise=rnd_noise[:, 0o300:0o400],
            batch_size=batch_size,
        )
        full_mu = np.c_[rich_out, muon_out, gpidmu_out]
        full_h = np.c_[
            rich_out,
            np.full_like(muon_out, MUONLL_ERRORCODE),
            gpidh_out,
            np.full(len(rich_out), PROBMU_ERRORCODE),
        ]
        full_out = np.where(ismuon_flag > 0.5, full_mu, full_h)
        return full_out
