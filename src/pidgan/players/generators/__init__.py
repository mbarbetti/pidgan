import keras as k

v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]

if v_major == 3 and v_minor >= 0:
    from .k3.Generator import Generator
    from .k3.ResGenerator import ResGenerator
else:
    from .k2.Generator import Generator
    from .k2.ResGenerator import ResGenerator
