import pprint

import yaml


class HPSingleton:
    def __init__(self, **kwargs) -> None:
        self._hparams = dict(**kwargs)
        self._used_keys = set()

    def update(self, **kwargs) -> None:
        for key in kwargs.keys():
            if key in self._used_keys:
                raise KeyError(
                    f"The hyperparameter {key} was already used and is now read-only"
                )
        self._hparams.update(kwargs)

    def get(self, key, value):
        if key not in self._hparams.keys():
            self._hparams[key] = value
        self._used_keys.add(key)
        return self._hparams[key]

    def clean(self) -> None:
        self._hparams = dict()
        self._used_keys = set()

    def __del__(self) -> None:
        for key in self._hparams.keys():
            if key not in self._used_keys:
                print(f"[WARNING] The hyperparameter {key} was defined but never used")
                print(self._used_keys)

    def __str__(self) -> str:
        return pprint.pformat(self._hparams)

    def get_dict(self) -> dict:
        return dict(**self._hparams)

    def dump(self, filename) -> None:
        with open(filename, "w") as file:
            yaml.dump(self.get_dict(), file)


__HPARAMS__ = None


def initHPSingleton() -> HPSingleton:
    global __HPARAMS__
    if __HPARAMS__ is None:
        __HPARAMS__ = HPSingleton()
    return __HPARAMS__
