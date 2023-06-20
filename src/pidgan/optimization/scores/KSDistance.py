import numpy as np

from pidgan.optimization.scores.BaseScore import BaseScore


class KSDistance(BaseScore):
    def __init__(self, name="kst_score", dtype=None) -> None:
        super().__init__(name, dtype)

    def __call__(
        self,
        x_true,
        x_pred,
        bins=10,
        range=None,
        weights_true=None,
        weights_pred=None,
        min_entries=100,
    ) -> float:
        if len(x_true) < min_entries:
            return None
        else:
            pdf_true, bins_ = np.histogram(
                x_true, bins=bins, range=range, weights=weights_true
            )
            pdf_pred, _ = np.histogram(
                x_pred, bins=bins_, range=None, weights=weights_pred
            )

            cdf_true = np.cumsum(pdf_true).astype(self._dtype)
            cdf_true /= cdf_true[-1] + 1e-12

            cdf_pred = np.cumsum(pdf_pred).astype(self._dtype)
            cdf_pred /= cdf_pred[-1] + 1e-12

            score = np.max(np.abs(cdf_true - cdf_pred))
            return float(score)
