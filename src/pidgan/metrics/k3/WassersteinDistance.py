import keras as k

from pidgan.metrics.k3.BaseMetric import BaseMetric


class WassersteinDistance(BaseMetric):
    def __init__(self, name="wass_dist", dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        if sample_weight is not None:
            state = k.ops.sum(sample_weight * (y_pred - y_true))
            state /= k.ops.sum(sample_weight)
        else:
            state = k.ops.mean(y_pred - y_true)
        state = k.ops.cast(state, self.dtype)
        print("debug:", self.dtype)
        self._metric_values.assign(state)
