import keras as k

k_vrs = k.__version__.split(".")[:2]
k_vrs = float(".".join([n for n in k_vrs]))

if k_vrs >= 3.0:
    from .k3.AuxDiscriminator import AuxDiscriminator
    from .k3.Discriminator import Discriminator
    from .k3.ResDiscriminator import ResDiscriminator
else:
    from .k2.AuxDiscriminator import AuxDiscriminator
    from .k2.Discriminator import Discriminator
    from .k2.ResDiscriminator import ResDiscriminator
