import tensorflow as tf
from tensorflow import keras

K = keras.backend


class LearnRateBaseScheduler(keras.callbacks.Callback):
    def __init__(self, optimizer, verbose=False, key="lr") -> None:
        super().__init__()
        self._name = "LearnRateBaseScheduler"

        # Optimizer
        assert isinstance(optimizer, keras.optimizers.Optimizer)
        self._optimizer = optimizer

        # Verbose
        assert isinstance(verbose, bool)
        self._verbose = verbose

        # Key name
        assert isinstance(key, str)
        self._key = key

    def on_train_begin(self, logs=None) -> None:
        init_lr = K.get_value(self._optimizer.learning_rate)
        self._init_lr = tf.identity(init_lr)
        self._dtype = self._init_lr.dtype
        self._step = -1

    def on_batch_begin(self, batch, logs=None) -> None:
        self._step += 1
        step = tf.cast(self._step, self._dtype)
        K.set_value(
            self._optimizer.learning_rate, self._scheduled_lr(self._init_lr, step)
        )

    def _scheduled_lr(self, init_lr, step) -> tf.Tensor:
        return init_lr

    def on_batch_end(self, batch, logs=None) -> None:
        logs = logs or {}
        if self._verbose:
            logs[self._key] = K.get_value(self._optimizer.learning_rate)

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        if self._verbose:
            logs[self._key] = K.get_value(self._optimizer.learning_rate)

    @property
    def name(self) -> str:
        return self._name

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        return self._optimizer

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def key(self) -> str:
        return self._key
