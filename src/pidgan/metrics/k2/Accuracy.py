import keras as k
import tensorflow as tf

from pidgan.metrics.k2.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name="accuracy", dtype=None, threshold=0.5) -> None:
        super().__init__(name=name, dtype=dtype)
        self._accuracy = k.metrics.BinaryAccuracy(
            name=name, dtype=dtype, threshold=threshold
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._accuracy(
            tf.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        self._metric_values.assign(state)
