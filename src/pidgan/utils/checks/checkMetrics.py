from pidgan.metrics import Accuracy
from pidgan.metrics import BinaryCrossentropy as BCE
from pidgan.metrics import JSDivergence as JS_div
from pidgan.metrics import KLDivergence as KL_div
from pidgan.metrics import MeanAbsoluteError as MAE
from pidgan.metrics import MeanSquaredError as MSE
from pidgan.metrics import RootMeanSquaredError as RMSE
from pidgan.metrics import WassersteinDistance as Wass_dist
from pidgan.metrics.BaseMetric import BaseMetric

METRIC_SHORTCUTS = [
    "accuracy",
    "bce",
    "kl_div",
    "js_div",
    "mse",
    "rmse",
    "mae",
    "wass_dist",
]
PIDGAN_METRICS = [
    Accuracy(),
    BCE(),
    KL_div(),
    JS_div(),
    MSE(),
    RMSE(),
    MAE(),
    Wass_dist(),
]


def checkMetrics(metrics):  # TODO: add Union[list, None]
    if metrics is None:
        return None
    else:
        if isinstance(metrics, list):
            checked_metrics = list()
            for metric in metrics:
                if isinstance(metric, str):
                    if metric in METRIC_SHORTCUTS:
                        for str_metric, calo_metric in zip(
                            METRIC_SHORTCUTS, PIDGAN_METRICS
                        ):
                            if metric == str_metric:
                                checked_metrics.append(calo_metric)
                    else:
                        raise ValueError(
                            f"`metrics` elements should be selected in "
                            f"{METRIC_SHORTCUTS}, instead '{metric}' passed"
                        )
                elif isinstance(metric, BaseMetric):
                    checked_metrics.append(metric)
                else:
                    raise TypeError(
                        f"`metrics` elements should be a pidgan's "
                        f"`BaseMetric`, instead {type(metric)} passed"
                    )
            return checked_metrics
        else:
            raise TypeError(
                f"`metrics` should be a list of strings or pidgan's "
                f"`BaseMetric`s, instead {type(metrics)} passed"
            )
