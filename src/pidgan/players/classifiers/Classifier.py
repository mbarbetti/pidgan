import tensorflow as tf

from pidgan.players.discriminators import Discriminator


class Classifier(Discriminator):
    def __init__(
        self,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_units=mlp_hidden_units,
            mlp_dropout_rates=mlp_dropout_rates,
            output_activation="sigmoid",
            name=name,
            dtype=dtype,
        )

    def hidden_feature(self, x) -> tf.Tensor:
        raise NotImplementedError(
            "Only the `discriminators` family has the "
            "`hidden_feature()` method implemented."
        )
