import tensorflow as tf

from pidgan.callbacks.schedulers.LearnRateBaseScheduler import LearnRateBaseScheduler


class LearnRatePolynomialDecay(LearnRateBaseScheduler):
    def __init__(
        self,
        optimizer,
        decay_steps,
        end_learning_rate=0.0001,
        power=1.0,
        cycle=False,
        verbose=False,
        key="lr",
    ) -> None:
        super().__init__(optimizer, verbose, key)
        self._name = "LearnRatePolynomialDecay"

        # Decay steps
        assert isinstance(decay_steps, (int, float))
        assert decay_steps >= 1
        self._decay_steps = int(decay_steps)

        # End learning-rate
        assert isinstance(end_learning_rate, (int, float))
        assert end_learning_rate > 0.0
        self._end_learning_rate = float(end_learning_rate)

        # Power
        assert isinstance(power, (int, float))
        assert power > 0.0
        self._power = float(power)

        # Cycle
        assert isinstance(cycle, bool)
        self._cycle = cycle

    def on_train_begin(self, logs=None) -> None:
        super().on_train_begin(logs=logs)
        self._tf_decay_steps = tf.cast(self._decay_steps, self._dtype)
        self._tf_end_learning_rate = tf.cast(self._end_learning_rate, self._dtype)
        self._tf_power = tf.cast(self._power, self._dtype)

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
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
