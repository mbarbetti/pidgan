from tensorflow import keras
from pidgan.players.generators.Generator import Generator

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

        # Model hidden layers
        self._hidden_layers = list()
        for i in range(self._num_hidden_layers):
            seq = list()
            seq.append(
                keras.layers.Dense(
                    units=self._mlp_hidden_units,
                    activation=None,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            seq.append(
                keras.layers.LeakyReLU(
                    alpha=LEAKY_ALPHA, name=f"leaky_relu_{i}" if name else None
                )
            )
            seq.append(
                keras.layers.Dropout(
                    rate=self._mlp_dropout_rates, name=f"dropout_{i}" if name else None
                )
            )
            self._hidden_layers.append(seq)

        # Model add layers
        self._add_layers = list()
        for i in range(self._num_hidden_layers - 1):
            self._add_layers.append(
                keras.layers.Add(name=f"add_{i}-{i+1}" if name else None)
            )

        # Model output layer
        self._out = keras.layers.Dense(
            units=output_dim,
            activation=output_activation,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="dense_out" if name else None,
            dtype=self.dtype,
        )

    def _build_model(self, x) -> None:
        if self._model is None:
            inputs = keras.layers.Input(shape=x.shape[1:])
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
            self._model = keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=f"{self.name}_model" if self.name else None,
            )
        else:
            pass

    @property
    def mlp_hidden_units(self) -> int:
        return self._mlp_hidden_units

    @property
    def mlp_dropout_rates(self) -> float:
        return self._mlp_dropout_rates

    @property
    def export_model(self) -> keras.Model:
        return self._model
