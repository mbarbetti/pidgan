import keras as k
from pidgan.players.discriminators import Discriminator


class MultiClassifier(Discriminator):
    def __init__(
        self,
        num_multiclasses,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_hidden_activation="leaky_relu",
        mlp_hidden_kernel_regularizer=None,
        mlp_dropout_rates=0.0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_dim=num_multiclasses,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_units=mlp_hidden_units,
            mlp_dropout_rates=mlp_dropout_rates,
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
        if isinstance(input_shape, (tuple, list)):
            return super()._get_input_dim(input_shape)
        else:
            return input_shape[-1]

    def build(self, input_shape) -> None:
        super().build(input_shape=input_shape)
        self._model.add(k.layers.Softmax(name="softmax_out" if self.name else None))

    def hidden_feature(self, x, return_hidden_idx=False):
        raise NotImplementedError(
            "Only the `discriminators` family has the "
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
