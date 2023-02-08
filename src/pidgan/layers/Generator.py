import tensorflow as tf


class Generator(tf.keras.layers.Layer):
    def __init__(
        self,
        name=None,
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
