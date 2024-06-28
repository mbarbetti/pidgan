import keras as k

v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]

if v_major == 3 and v_minor >= 0:
    from .k3.AuxDiscriminator import AuxDiscriminator
    from .k3.Discriminator import Discriminator
    from .k3.ResDiscriminator import ResDiscriminator
else:
    from .k2.AuxDiscriminator import AuxDiscriminator
    from .k2.Discriminator import Discriminator
    from .k2.ResDiscriminator import ResDiscriminator
