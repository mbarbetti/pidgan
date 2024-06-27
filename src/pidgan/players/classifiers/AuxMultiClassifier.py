import keras as k
from pidgan.players.discriminators import AuxDiscriminator


class AuxMultiClassifier(AuxDiscriminator):
    def __init__(
        self,
        num_multiclasses,
        aux_features,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_hidden_activation="leaky_relu",
        mlp_hidden_kernel_regularizer=None,
        mlp_dropout_rates=0.0,
        enable_residual_blocks=False,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_dim=num_multiclasses,
            aux_features=aux_features,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_units=mlp_hidden_units,
            mlp_dropout_rates=mlp_dropout_rates,
            enable_residual_blocks=enable_residual_blocks,
            output_activation=None,
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
                in_dim = in_shape_2
            else:
                in_dim = in_shape_1[-1] + in_shape_2[-1]
        else:
            in_dim = input_shape[-1]  # after concatenate action
        in_dim += len(self._aux_features)
        return in_dim

    def _define_arch(self) -> None:
        super()._define_arch()
        self._out.append(k.layers.Softmax(name="softmax_out" if self.name else None))

    def hidden_feature(self, x, return_hidden_idx=False):
        raise NotImplementedError(
            "Only the pidgan's Discriminators has the "
            "`hidden_feature()` method implemented."
        )

    @property
    def num_multiclasses(self) -> int:
        return self._output_dim

    @property
    def mlp_hidden_activation(self):
        return self._hidden_activation_func

    @property
    def mlp_hidden_kernel_regularizer(self):
        return self._hidden_kernel_reg
