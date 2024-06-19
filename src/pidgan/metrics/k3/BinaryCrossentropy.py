import keras as k

from pidgan.metrics.k3.BaseMetric import BaseMetric


class BinaryCrossentropy(BaseMetric):
    def __init__(
        self, name="bce", dtype=None, from_logits=False, label_smoothing=0.0
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        self._bce = k.metrics.BinaryCrossentropy(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._bce(k.ops.ones_like(y_pred), y_pred, sample_weight=sample_weight)
        self._metric_values.assign(state)
