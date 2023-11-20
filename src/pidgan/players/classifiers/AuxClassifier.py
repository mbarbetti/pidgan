from pidgan.players.discriminators import AuxDiscriminator


class AuxClassifier(AuxDiscriminator):
    def __init__(
        self,
        aux_features,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        enable_residual_blocks=False,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_dim=1,
            aux_features=aux_features,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_units=mlp_hidden_units,
            mlp_dropout_rates=mlp_dropout_rates,
            enable_residual_blocks=enable_residual_blocks,
            output_activation="sigmoid",
            name=name,
            dtype=dtype,
        )

    def hidden_feature(self, x, return_hidden_idx=False):
        raise NotImplementedError(
            "Only the `discriminators` family has the "
            "`hidden_feature()` method implemented."
        )
