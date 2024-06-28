import keras as k


class BaseMetric(k.metrics.Metric):
    def __init__(self, name="metric", dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)
        self._metric_values = self.add_weight(
            name=f"{name}_values", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        raise NotImplementedError(
            "Only the pidgan's BaseMetric subclasses have the "
            "`update_state()` method implemented."
        )

    def result(self):
        return self._metric_values.value

    def reset_state(self):
        self._metric_values.assign(0.0)
