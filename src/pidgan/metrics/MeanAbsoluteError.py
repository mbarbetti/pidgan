from tensorflow import keras

from pidgan.metrics.BaseMetric import BaseMetric


class MeanAbsoluteError(BaseMetric):
    def __init__(self, name="mae", dtype=None, **kwargs) -> None:
        super().__init__(name, dtype, **kwargs)
        self._mae = keras.metrics.MeanAbsoluteError(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._mae(y_true, y_pred, sample_weight=sample_weight)
        self._metric_values.assign(state)
