import pickle

import numpy as np
from tensorflow import keras

MUON_ERRORCODE = -1000.0


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
        self._gpid = ml_pipes["GlobalPID"]
        self._gmuid = ml_pipes["GlobalMuonId"]

    def predict(self, x, rnd_noise, ismuon_flag, batch_size=None):
        rich_out = self._rich.predict(
            x, rnd_noise=rnd_noise[:, 0o000:0o100], batch_size=batch_size
        )
        muon_out = self._muon.predict(
            x, rnd_noise=rnd_noise[:, 0o100:0o200], batch_size=batch_size
        )
        muon_out = np.where(ismuon_flag > 0.0, muon_out, MUON_ERRORCODE)
        gpid_out = self._gpid.predict(
            np.concatenate([x, rich_out, ismuon_flag, muon_out], axis=1),
            rnd_noise=rnd_noise[:, 0o200:0o300],
            batch_size=batch_size,
        )
        gmuid_out = self._gmuid.predict(
            np.concatenate([x, rich_out, muon_out], axis=1),
            rnd_noise=rnd_noise[:, 0o300:0o400],
            batch_size=batch_size,
        )
        full_out = np.concatenate([rich_out, muon_out, gpid_out, gmuid_out], axis=1)
        return full_out
