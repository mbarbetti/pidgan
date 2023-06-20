import numpy as np

from pidgan.optimization.scores.BaseScore import BaseScore


class EMDistance(BaseScore):
    def __init__(self, name="emd_score", dtype=None) -> None:
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

            pdf_true = pdf_true.astype(self.dtype)
            pdf_true /= np.sum(pdf_true) + 1e-12

            pdf_pred = pdf_pred.astype(self.dtype)
            pdf_pred /= np.sum(pdf_pred) + 1e-12

            emd_scores = np.zeros(shape=(len(bins_)))
            for i in np.arange(len(emd_scores) - 1):
                emd_scores[i + 1] = pdf_true[i] + emd_scores[i] - pdf_pred[i]

            score = np.mean(np.abs(emd_scores))
            return float(score)
