from tensorflow import keras

from pidgan.metrics.BaseMetric import BaseMetric


class RootMeanSquaredError(BaseMetric):
    def __init__(self, name="rmse", dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self._rmse = keras.metrics.RootMeanSquaredError(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        state = self._rmse(y_true, y_pred, sample_weight=sample_weight)
        self._metric_values.assign(state)
