import keras as k

v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]

if v_major == 3 and v_minor >= 0:
    from .k3.BceGAN import BceGAN
    from .k3.BceGAN_ALP import BceGAN_ALP
    from .k3.BceGAN_GP import BceGAN_GP
    from .k3.CramerGAN import CramerGAN
    from .k3.GAN import GAN
    from .k3.LSGAN import LSGAN
    from .k3.WGAN import WGAN
    from .k3.WGAN_ALP import WGAN_ALP
    from .k3.WGAN_GP import WGAN_GP
else:
    from .k2.BceGAN import BceGAN
    from .k2.BceGAN_ALP import BceGAN_ALP
    from .k2.BceGAN_GP import BceGAN_GP
    from .k2.CramerGAN import CramerGAN
    from .k2.GAN import GAN
    from .k2.LSGAN import LSGAN
    from .k2.WGAN import WGAN
    from .k2.WGAN_ALP import WGAN_ALP
    from .k2.WGAN_GP import WGAN_GP
