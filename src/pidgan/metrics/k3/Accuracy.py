import keras as k

from pidgan.metrics.k3.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name="accuracy", dtype=None, threshold=0.5) -> None:
        super().__init__(name=name, dtype=dtype)
        self._accuracy = k.metrics.BinaryAccuracy(
            name=name, dtype=dtype, threshold=threshold
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._accuracy(
            k.ops.ones_like(y_pred), y_pred, sample_weight=sample_weight
        )
        self._metric_values.assign(state)
