import tensorflow as tf

from pidgan.callbacks.schedulers.LearnRateBaseScheduler import LearnRateBaseScheduler


class LearnRateExpDecay(LearnRateBaseScheduler):
    def __init__(
        self,
        optimizer,
        decay_rate,
        decay_steps,
        staircase=False,
        min_learning_rate=None,
        verbose=False,
        key="lr",
    ) -> None:
        super().__init__(optimizer, verbose, key)
        self._name = "LearnRateExpDecay"

        # Decay rate
        assert isinstance(decay_rate, (int, float))
        assert decay_rate > 0.0
        self._decay_rate = float(decay_rate)

        # Decay steps
        assert isinstance(decay_steps, (int, float))
        assert decay_steps >= 1
        self._decay_steps = int(decay_steps)

        # Staircase
        assert isinstance(staircase, bool)
        self._staircase = staircase

        # Minimum learning-rate
        if min_learning_rate is not None:
            assert isinstance(min_learning_rate, (int, float))
            assert min_learning_rate > 0.0
            self._min_learning_rate = float(min_learning_rate)
        else:
            self._min_learning_rate = None

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_decay_rate = tf.cast(self._decay_rate, self._dtype)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
        p = tf.divide(step, self._tf_decay_steps)
        if self._staircase:
            p = tf.floor(p)
        sched_lr = tf.multiply(init_lr, tf.pow(self._tf_decay_rate, p))
        if self._min_learning_rate is not None:
            return tf.maximum(sched_lr, self._min_learning_rate)
        else:
            return sched_lr

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def decay_steps(self) -> int:
        return self._decay_steps

    @property
    def staircase(self) -> bool:
        return self._staircase

    @property
    def min_learning_rate(self) -> float:
        return self._min_learning_rate
