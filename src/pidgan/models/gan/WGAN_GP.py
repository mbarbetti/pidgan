import tensorflow as tf
from .GAN import GAN


class WGAN_GP(GAN):
    def __init__(
        self,
        name=None,
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
