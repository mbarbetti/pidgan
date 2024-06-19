import keras as k

k_vrs = k.__version__.split(".")[:2]
k_vrs = float(".".join([n for n in k_vrs]))

if k_vrs >= 3.0:
    from .k3.Generator import Generator
    from .k3.ResGenerator import ResGenerator
else:
    from .k2.Generator import Generator
    from .k2.ResGenerator import ResGenerator
