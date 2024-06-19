import keras

from pidgan.metrics.k2.BaseMetric import BaseMetric


class MeanSquaredError(BaseMetric):
    def __init__(self, name="mse", dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)
        self._mse = keras.metrics.MeanSquaredError(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._mse(y_true, y_pred, sample_weight=sample_weight)
        self._metric_values.assign(state)
