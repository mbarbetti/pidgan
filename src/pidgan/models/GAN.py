import tensorflow as tf
from pidgan.layers import Generator, Discriminator, Referee


class GAN(tf.keras.Model):
    def __init__(
        self,
        name=None,
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
