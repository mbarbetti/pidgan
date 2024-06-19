import keras as k

k_vrs = k.__version__.split(".")[:2]
k_vrs = float(".".join([n for n in k_vrs]))

if k_vrs >= 3.0:
    from .k3.LearnRateBaseScheduler import LearnRateBaseScheduler
    from .k3.LearnRateCosineDecay import LearnRateCosineDecay
    from .k3.LearnRateExpDecay import LearnRateExpDecay
    from .k3.LearnRateInvTimeDecay import LearnRateInvTimeDecay
    from .k3.LearnRatePiecewiseConstDecay import LearnRatePiecewiseConstDecay
    from .k3.LearnRatePolynomialDecay import LearnRatePolynomialDecay
else:
    from .k2.LearnRateBaseScheduler import LearnRateBaseScheduler
    from .k2.LearnRateCosineDecay import LearnRateCosineDecay
    from .k2.LearnRateExpDecay import LearnRateExpDecay
    from .k2.LearnRateInvTimeDecay import LearnRateInvTimeDecay
    from .k2.LearnRatePiecewiseConstDecay import LearnRatePiecewiseConstDecay
    from .k2.LearnRatePolynomialDecay import LearnRatePolynomialDecay
