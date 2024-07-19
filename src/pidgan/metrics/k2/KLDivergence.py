import keras as k

from pidgan.metrics.k2.BaseMetric import BaseMetric


class KLDivergence(BaseMetric):
    def __init__(self, name="kl_div", dtype=None) -> None:
        super().__init__(name=name, dtype=dtype)
        self._kl_div = k.metrics.KLDivergence(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        state = self._kl_div(y_true, y_pred, sample_weight=sample_weight)
        self._metric_values.assign(state)
