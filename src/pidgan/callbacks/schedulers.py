import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

K = tf.keras.backend


class BaseScheduler(Callback):
    def __init__(self, optimizer, verbose=False):
        super().__init__()
        self._optimizer = optimizer
        self._verbose = bool(verbose)

    def on_train_begin(self, logs=None):
        init_lr = K.get_value(self._optimizer.learning_rate)
        self._init_lr = tf.identity(init_lr)
        self._dtype = self._init_lr.dtype
        self._step = -1

    def on_batch_begin(self, batch, logs=None):
        self._step += 1
        step = tf.cast(self._step, self._dtype)
        K.set_value(
            self._optimizer.learning_rate, self._scheduled_lr(self._init_lr, step)
        )

    def _scheduled_lr(self, init_lr, step):
        return init_lr

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self._verbose:
            logs["lr"] = K.get_value(self.model.optimizer.learning_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._verbose:
            logs["lr"] = K.get_value(self.model.optimizer.learning_rate)

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return self._optimizer


class CosineDecay(BaseScheduler):
    def __init__(self, optimizer, decay_steps, alpha=0.0, verbose=False):
        super().__init__(optimizer, verbose)
        assert decay_steps > 0
        self._decay_steps = int(decay_steps)
        assert (alpha) >= 0.0 and (alpha <= 1.0)
        self._alpha = float(alpha)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)
        self._tf_alpha = tf.cast(self._alpha, self._dtype)

    def _scheduled_lr(self, init_lr, step):
        step = tf.minimum(step, self._tf_decay_steps)
        p = tf.divide(step, self._tf_decay_steps)
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * p))
        decayed = tf.multiply(1 - self._tf_alpha, cosine_decay + self._tf_alpha)
        return tf.multiply(init_lr, decayed)

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def alpha(self) -> float:
        return self._alpha


class ExponentialDecay(BaseScheduler):
    def __init__(
        self, optimizer, decay_rate, decay_steps, staircase=False, verbose=False
    ):
        super().__init__(optimizer, verbose)
        assert decay_rate > 0.0
        self._decay_rate = float(decay_rate)
        assert decay_steps > 0
        self._decay_steps = int(decay_steps)
        assert isinstance(staircase, bool)
        self._staircase = staircase

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self._tf_decay_rate = tf.cast(self._decay_rate, self._dtype)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)

    def _scheduled_lr(self, init_lr, step):
        p = tf.divide(step, self._tf_decay_steps)
        if self._staircase:
            p = tf.floor(p)
        return tf.multiply(init_lr, tf.pow(self._tf_decay_rate, p))

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def staircase(self) -> bool:
        return self._staircase


class InverseTimeDecay(BaseScheduler):
    def __init__(
        self, optimizer, decay_rate, decay_steps, staircase=False, verbose=False
    ):
        super().__init__(optimizer, verbose)
        assert decay_rate > 0.0
        self._decay_rate = float(decay_rate)
        assert decay_steps > 0
        self._decay_steps = int(decay_steps)
        assert isinstance(staircase, bool)
        self._staircase = staircase

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self._tf_decay_rate = tf.cast(self._decay_rate, self._dtype)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)

    def _scheduled_lr(self, init_lr, step):
        p = tf.divide(step, self._tf_decay_steps)
        if self._staircase:
            p = tf.floor(p)
        return tf.divide(init_lr, 1 + tf.multiply(self._tf_decay_rate, p))

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def staircase(self) -> bool:
        return self._staircase


class PiecewiseConstantDecay(BaseScheduler):
    def __init__(self, optimizer, boundaries, values, verbose=False):
        super().__init__(optimizer, verbose)
        assert isinstance(boundaries, (list, tuple, np.ndarray))
        assert isinstance(values, (list, tuple, np.ndarray))
        assert len(boundaries) >= 1
        assert len(values) >= 2
        assert len(boundaries) == len(values) - 1
        self._boundaries = [0] + [int(b) for b in boundaries]
        self._values = [float(v) for v in values]

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self._tf_boundaries = tf.cast(self._boundaries, self._dtype)
        self._tf_values = tf.cast(self._values, self._dtype)

    def _scheduled_lr(self, init_lr, step):
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


class PolynomialDecay(BaseScheduler):
    def __init__(
        self,
        optimizer,
        decay_steps,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False,
        verbose=False,
    ):
        super().__init__(optimizer, verbose)
        assert decay_steps > 0
        self._decay_steps = int(decay_steps)
        assert end_learning_rate > 0.0
        self._end_learning_rate = float(end_learning_rate)
        assert power > 0.0
        self._power = float(power)
        assert isinstance(cycle, bool)
        self._cycle = cycle

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs=logs)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)
        self._tf_end_learning_rate = tf.cast(self._end_learning_rate, self._dtype)
        self._tf_power = tf.cast(self._power, self._dtype)

    def _scheduled_lr(self, init_lr, step):
        if not self._cycle:
            step = tf.minimum(step, self._tf_decay_steps)
            decay_steps = self._tf_decay_steps
        else:
            decay_steps = tf.multiply(
                self._tf_decay_steps,
                tf.math.ceil(tf.divide(step, self._tf_decay_steps)),
            )
        return (
            (init_lr - self._tf_end_learning_rate)
            * tf.pow(1 - step / decay_steps, self._tf_power)
        ) + self._tf_end_learning_rate

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def end_learning_rate(self) -> float:
        return self._end_learning_rate

    @property
    def power(self) -> float:
        return self._power

    @property
    def cycle(self) -> bool:
        return self._cycle
