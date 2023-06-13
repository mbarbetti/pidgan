import numpy as np
import tensorflow as tf

from pidgan.callbacks.schedulers.LearnRateBaseScheduler import LearnRateBaseScheduler


class LearnRateCosineDecay(LearnRateBaseScheduler):
    def __init__(
        self,
        optimizer,
        decay_steps,
        alpha=0.0,
        min_learning_rate=None,
        verbose=False,
        key="lr",
    ) -> None:
        super().__init__(optimizer, verbose, key)
        self._name = "LearnRateCosineDecay"

        # Decay steps
        assert isinstance(decay_steps, (int, float))
        assert decay_steps >= 1
        self._decay_steps = int(decay_steps)

        # Alpha
        assert isinstance(alpha, (int, float))
        assert (alpha) >= 0.0 and (alpha <= 1.0)
        self._alpha = float(alpha)

        # Minimum learning-rate
        if min_learning_rate is not None:
            assert isinstance(min_learning_rate, (int, float))
            assert min_learning_rate > 0.0
            self._min_learning_rate = float(min_learning_rate)
        else:
            self._min_learning_rate = None

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)
        self._tf_alpha = tf.cast(self._alpha, self._dtype)

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
        step = tf.minimum(step, self._tf_decay_steps)
        p = tf.divide(step, self._tf_decay_steps)
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(np.pi) * p))
        decayed = tf.multiply(1 - self._tf_alpha, cosine_decay + self._tf_alpha)
        sched_lr = tf.multiply(init_lr, decayed)
        if self._min_learning_rate is not None:
            return tf.maximum(sched_lr, self._min_learning_rate)
        else:
            return sched_lr

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def min_learning_rate(self) -> float:
        return self._min_learning_rate
