import tensorflow as tf
from tensorflow import keras

LEAKY_ALPHA = 0.1


class Discriminator(keras.Model):
    def __init__(
        self,
        output_dim,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        dropout_rate=0.0,
        output_activation="sigmoid",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

        # Output dimension
        assert output_dim >= 1
        self._output_dim = int(output_dim)

        # Number of hidden layers
        assert isinstance(num_hidden_layers, (int, float))
        assert num_hidden_layers >= 1
        self._num_hidden_layers = int(num_hidden_layers)

        # Multilayer perceptron hidden units
        if isinstance(mlp_hidden_units, (int, float)):
            assert mlp_hidden_units >= 1
            self._mlp_hidden_units = [int(mlp_hidden_units)] * self._num_hidden_layers
        else:
            mlp_hidden_units = list(mlp_hidden_units)
            assert len(mlp_hidden_units) == self._num_hidden_layers
            self._mlp_hidden_units = list()
            for units in mlp_hidden_units:
                assert isinstance(units, (int, float))
                assert mlp_hidden_units >= 1
                self._mlp_hidden_units.append(int(units))

        # Dropout rate
        if isinstance(dropout_rate, (int, float)):
            assert dropout_rate >= 0.0 and dropout_rate < 1.0
            self._dropout_rate = [float(dropout_rate)] * self._num_hidden_layers
        else:
            dropout_rate = list(dropout_rate)
            assert len(dropout_rate) == self._num_hidden_layers
            self._dropout_rate = list()
            for rate in dropout_rate:
                assert isinstance(rate, (int, float))
                assert rate >= 0.0 and rate < 1.0
                self._dropout_rate.append(float(rate))

        # Output activation
        self._output_activation = output_activation

        # Model
        self._seq = keras.Sequential(name=f"{name}_seq" if name else None)
        for i, (units, rate) in enumerate(
            zip(self._mlp_hidden_units, self._dropout_rate)
        ):
            self._seq.add(
                keras.layers.Dense(
                    units=units,
                    activation=None,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            self._seq.add(
                keras.layers.LeakyReLU(
                    alpha=LEAKY_ALPHA, name=f"leaky_relu_{i}" if name else None
                )
            )
            self._seq.add(
                keras.layers.Dropout(rate=rate, name=f"dropout_{i}" if name else None)
            )
        self._seq.add(
            keras.layers.Dense(
                units=output_dim,
                activation=output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="dense_out" if name else None,
                dtype=self.dtype,
            )
        )

    def _prepare_input(self, inputs) -> tf.Tensor:
        d_in = tf.concat(inputs, axis=-1)
        return d_in

    def call(self, inputs) -> tf.Tensor:
        d_in = self._prepare_input(inputs)
        out = self._seq(d_in)
        return out

    def summary(self, **kwargs) -> None:
        self._seq.summary(**kwargs)

    def hidden_feature(self, inputs, return_hidden_idx=False):
        hidden_idx = int((self._num_hidden_layers + 1) / 2.0)
        if hidden_idx < 1:
            hidden_idx += 1
        x = self._prepare_input(inputs)
        for layer in self._seq.layers[: 3 * hidden_idx]:  # dense + relu + dropout
            x = layer(x)
        if return_hidden_idx:
            return x, hidden_idx
        else:
            return x

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_hidden_layers(self) -> int:
        return self._num_hidden_layers

    @property
    def mlp_hidden_units(self) -> list:
        return self._mlp_hidden_units

    @property
    def dropout_rate(self) -> list:
        return self._dropout_rate

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def export_model(self) -> keras.Sequential:
        return self._seq
