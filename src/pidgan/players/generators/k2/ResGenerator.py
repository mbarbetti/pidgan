import keras as k

from pidgan.players.generators.k2.Generator import Generator

LEAKY_ALPHA = 0.1


class ResGenerator(Generator):
    def __init__(
        self,
        output_dim,
        latent_dim,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        output_activation=None,
        name=None,
        dtype=None,
    ) -> None:
        super(Generator, self).__init__(name=name, dtype=dtype)
        self._hidden_activation_func = None
        self._enable_res_blocks = True
        self._model = None

        # Output dimension
        assert output_dim >= 1
        self._output_dim = int(output_dim)

        # Latent space dimension
        assert latent_dim >= 1
        self._latent_dim = int(latent_dim)

        # Number of hidden layers
        assert isinstance(num_hidden_layers, (int, float))
        assert num_hidden_layers >= 1
        self._num_hidden_layers = int(num_hidden_layers)

        # Multilayer perceptron hidden units
        assert isinstance(mlp_hidden_units, (int, float))
        assert mlp_hidden_units >= 1
        self._mlp_hidden_units = int(mlp_hidden_units)

        # Dropout rate
        assert isinstance(mlp_dropout_rates, (int, float))
        assert mlp_dropout_rates >= 0.0 and mlp_dropout_rates < 1.0
        self._mlp_dropout_rates = float(mlp_dropout_rates)

        # Output activation
        self._output_activation = output_activation

    def _define_arch(self) -> None:
        self._hidden_layers = list()
        for i in range(self._num_hidden_layers):
            res_block = list()
            res_block.append(
                k.layers.Dense(
                    units=self._mlp_hidden_units,
                    activation=self._hidden_activation_func,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if self.name else None,
                    dtype=self.dtype,
                )
            )
            if self._hidden_activation_func is None:
                res_block.append(
                    k.layers.LeakyReLU(
                        alpha=LEAKY_ALPHA, name=f"leaky_relu_{i}" if self.name else None
                    )
                )
            res_block.append(
                k.layers.Dropout(
                    rate=self._mlp_dropout_rates,
                    name=f"dropout_{i}" if self.name else None,
                )
            )
            self._hidden_layers.append(res_block)

        self._add_layers = list()
        for i in range(self._num_hidden_layers - 1):
            self._add_layers.append(
                k.layers.Add(name=f"add_{i}-{i+1}" if self.name else None)
            )

        self._out = k.layers.Dense(
            units=self._output_dim,
            activation=self._output_activation,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="dense_out" if self.name else None,
            dtype=self.dtype,
        )

    def build(self, input_shape) -> None:
        in_dim = self._get_input_dim(input_shape)
        self._define_arch()
        inputs = k.layers.Input(shape=(in_dim,))
        x_ = inputs
        for layer in self._hidden_layers[0]:
            x_ = layer(x_)
        for i in range(1, self._num_hidden_layers):
            h_ = x_
            for layer in self._hidden_layers[i]:
                h_ = layer(h_)
            if self._enable_res_blocks:
                x_ = self._add_layers[i - 1]([x_, h_])
            else:
                x_ = h_
        outputs = self._out(x_)
        self._model = k.Model(
            inputs=inputs,
            outputs=outputs,
            name=f"{self.name}_func" if self.name else None,
        )

    @property
    def mlp_hidden_units(self) -> int:
        return self._mlp_hidden_units

    @property
    def mlp_dropout_rates(self) -> float:
        return self._mlp_dropout_rates

    @property
    def export_model(self) -> k.Model:
        return self._model
