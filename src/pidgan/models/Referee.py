import tensorflow as tf
from .Discriminator import Discriminator


class Referee(Discriminator):
    def __init__(
        self,
        name=None,
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
