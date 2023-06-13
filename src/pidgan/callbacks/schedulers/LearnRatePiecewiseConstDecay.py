import numpy as np
import tensorflow as tf

from pidgan.callbacks.schedulers.LearnRateBaseScheduler import LearnRateBaseScheduler


class LearnRatePiecewiseConstDecay(LearnRateBaseScheduler):
    def __init__(self, optimizer, boundaries, values, verbose=False, key="lr") -> None:
        super().__init__(optimizer, verbose, key)
        self._name = "LearnRatePiecewiseConstDecay"

        # Boundaries and values
        assert isinstance(boundaries, (list, tuple, np.ndarray))
        assert isinstance(values, (list, tuple, np.ndarray))
        assert len(boundaries) >= 1
        assert len(values) >= 2
        assert len(boundaries) == len(values) - 1
        self._boundaries = [0] + [int(b) for b in boundaries]
        self._values = [float(v) for v in values]

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_boundaries = tf.cast(self._boundaries, self._dtype)
        self._tf_values = tf.cast(self._values, self._dtype)

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
        for i in range(len(self._boundaries) - 1):
            if (step >= self._tf_boundaries[i]) and (step < self._tf_boundaries[i + 1]):
                return self._tf_values[i]
        return self._tf_values[-1]

    @property
    def boundaries(self) -> list:
        return self._boundaries

    @property
    def values(self) -> list:
        return self._values
