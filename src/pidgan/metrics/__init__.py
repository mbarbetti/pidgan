import keras as k

k_vrs = k.__version__.split(".")[:2]
k_vrs = float(".".join([n for n in k_vrs]))

if k_vrs >= 3.0:
    from .k3.BaseMetric import BaseMetric
    from .k3.Accuracy import Accuracy
    from .k3.BinaryCrossentropy import BinaryCrossentropy
    from .k3.JSDivergence import JSDivergence
    from .k3.KLDivergence import KLDivergence
    from .k3.MeanAbsoluteError import MeanAbsoluteError
    from .k3.MeanSquaredError import MeanSquaredError
    from .k3.RootMeanSquaredError import RootMeanSquaredError
    from .k3.WassersteinDistance import WassersteinDistance
else:
    from .k2.BaseMetric import BaseMetric
    from .k2.Accuracy import Accuracy
    from .k2.BinaryCrossentropy import BinaryCrossentropy
    from .k2.JSDivergence import JSDivergence
    from .k2.KLDivergence import KLDivergence
    from .k2.MeanAbsoluteError import MeanAbsoluteError
    from .k2.MeanSquaredError import MeanSquaredError
    from .k2.RootMeanSquaredError import RootMeanSquaredError
    from .k2.WassersteinDistance import WassersteinDistance
