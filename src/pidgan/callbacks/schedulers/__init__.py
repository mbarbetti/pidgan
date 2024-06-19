import keras as k

v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]

if v_major == 3 and v_minor >= 0:
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
