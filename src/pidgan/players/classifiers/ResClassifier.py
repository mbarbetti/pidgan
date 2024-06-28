from pidgan.players.discriminators import ResDiscriminator


class ResClassifier(ResDiscriminator):
    def __init__(
        self,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_hidden_activation="leaky_relu",
        mlp_hidden_kernel_regularizer=None,
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

        # Activation function
        if mlp_hidden_activation == "leaky_relu":
            self._hidden_activation_func = None
        else:
            self._hidden_activation_func = mlp_hidden_activation

        # Kernel regularizer
        self._hidden_kernel_reg = mlp_hidden_kernel_regularizer

    def _get_input_dim(self, input_shape) -> int:
        if isinstance(input_shape, (list, tuple)):
            in_shape_1, in_shape_2 = input_shape
            if isinstance(in_shape_2, int):
                return in_shape_2
            else:
                return in_shape_1[-1] + in_shape_2[-1]
        else:
            return input_shape[-1]  # after concatenate action

    def hidden_feature(self, x, return_hidden_idx=False):
        raise NotImplementedError(
            "Only the pidgan's Discriminators has the "
            "`hidden_feature()` method implemented."
        )

    @property
    def mlp_hidden_activation(self):
        return self._hidden_activation_func

    @property
    def mlp_hidden_kernel_regularizer(self):
        return self._hidden_kernel_reg
