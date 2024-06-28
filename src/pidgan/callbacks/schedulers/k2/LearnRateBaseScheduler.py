import keras as k
import tensorflow as tf

K = k.backend


class LearnRateBaseScheduler(k.callbacks.Callback):
    def __init__(self, optimizer, verbose=False, key="lr") -> None:
        super().__init__()
        self._name = "LearnRateBaseScheduler"

        # Optimizer
        assert isinstance(optimizer, k.optimizers.Optimizer)
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
        self._step = tf.cast(0.0, self._dtype)

    def on_batch_begin(self, batch, logs=None) -> None:
        self._step += 1.0
        sched_lr = self._scheduled_lr(self._init_lr, self._step)
        K.set_value(self._optimizer.learning_rate, sched_lr)

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
    def optimizer(self) -> k.optimizers.Optimizer:
        return self._optimizer

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def key(self) -> str:
        return self._key
