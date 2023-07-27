from tensorflow import keras


class HopaasPruner(keras.callbacks.Callback):
    def __init__(
        self, trial, loss_name, report_frequency=1, enable_pruning=True
    ) -> None:
        super().__init__()
        self._trial = trial
        assert isinstance(loss_name, str)
        self._loss_name = loss_name
        assert report_frequency >= 1
        self._report_freq = int(report_frequency)
        assert isinstance(enable_pruning, bool)
        self._enable_pruning = enable_pruning

    def on_epoch_end(self, epoch, logs=None) -> None:
        if self._enable_pruning:
            if (epoch + 1) % self._report_freq == 0:
                self._trial.loss = self._get_monitor_value(logs=logs)
                if self._trial.should_prune:
                    self.model.stop_training = True

    def on_train_end(self, logs=None) -> None:
        if not self._enable_pruning:
            self._trial.loss = self._get_monitor_value(logs=logs)

    def _get_monitor_value(self, logs) -> float:
        logs = logs or {}
        monitor_value = logs.get(self._loss_name)
        if monitor_value is None:
            raise ValueError(
                f"`loss_name` should be selected in {list(logs.keys())}, "
                f"instead '{self._loss_name}' passed"
            )
        return float(monitor_value)

    @property
    def trial(self):  # TODO: add Any
        return self._trial

    @property
    def loss_name(self) -> str:
        return self._loss_name

    @property
    def report_frequency(self) -> int:
        return self._report_freq

    @property
    def enable_pruning(self) -> bool:
        return self._enable_pruning
