import numpy as np

NP_DTYPE = np.float32


class BaseScore:
    def __init__(self, name="score", dtype=None) -> None:
        assert isinstance(name, str)
        self._name = name

        if dtype is not None:
            assert isinstance(dtype, type)
            self._dtype = dtype
        else:
            self._dtype = NP_DTYPE

    def __call__(self, x_true, x_pred, bins=10, range=None) -> float:
        raise NotImplementedError(
            "Only `BaseScore` subclasses have the `__call__()` method implemented."
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> type:
        return self._dtype
