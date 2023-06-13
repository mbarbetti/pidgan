import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError as TF_MSE

from pidgan.metrics.BaseMetric import BaseMetric


class MeanSquaredError(BaseMetric):
    def __init__(self, name="mse", dtype=None, **kwargs) -> None:
        super().__init__(name, dtype, **kwargs)
        self._mse = TF_MSE(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._mse(y_true, y_pred, sample_weight=sample_weight)
        self._metric_values.assign(state)
