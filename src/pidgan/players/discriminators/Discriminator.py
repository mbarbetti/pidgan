import tensorflow as tf
from tensorflow import keras

LEAKY_ALPHA = 0.1


class Discriminator(keras.Model):
    def __init__(
        self,
        output_dim,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
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
                assert units >= 1
                self._mlp_hidden_units.append(int(units))

        # Dropout rate
        if isinstance(mlp_dropout_rates, (int, float)):
            assert mlp_dropout_rates >= 0.0 and mlp_dropout_rates < 1.0
            self._mlp_dropout_rates = [
                float(mlp_dropout_rates)
            ] * self._num_hidden_layers
        else:
            mlp_dropout_rates = list(mlp_dropout_rates)
            assert len(mlp_dropout_rates) == self._num_hidden_layers
            self._mlp_dropout_rates = list()
            for rate in mlp_dropout_rates:
                assert isinstance(rate, (int, float))
                assert rate >= 0.0 and rate < 1.0
                self._mlp_dropout_rates.append(float(rate))

        # Output activation
        self._output_activation = output_activation

        # Model
        self._model = keras.Sequential(name=f"{name}_seq" if name else None)
        for i, (units, rate) in enumerate(
            zip(self._mlp_hidden_units, self._mlp_dropout_rates)
        ):
            self._model.add(
                keras.layers.Dense(
                    units=units,
                    activation=None,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            self._model.add(
                keras.layers.LeakyReLU(
                    alpha=LEAKY_ALPHA, name=f"leaky_relu_{i}" if name else None
                )
            )
            self._model.add(
                keras.layers.Dropout(rate=rate, name=f"dropout_{i}" if name else None)
            )
        self._model.add(
            keras.layers.Dense(
                units=output_dim,
                activation=output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="dense_out" if name else None,
                dtype=self.dtype,
            )
        )

    def _prepare_input(self, x) -> tf.Tensor:
        if isinstance(x, (list, tuple)):
            x = tf.concat(x, axis=-1)
        return x

    def _build_model(self, x) -> None:
        pass

    def call(self, x) -> tf.Tensor:
        x = self._prepare_input(x)
        self._build_model(x)
        out = self._model(x)
        return out

    def summary(self, **kwargs) -> None:
        self._model.summary(**kwargs)

    def hidden_feature(self, x, return_hidden_idx=False):
        hidden_idx = int((self._num_hidden_layers + 1) / 2.0)
        if hidden_idx < 1:
            hidden_idx += 1
        x = self._prepare_input(x)
        for layer in self._model.layers[: 3 * hidden_idx]:  # dense + relu + dropout
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
    def mlp_dropout_rates(self) -> list:
        return self._mlp_dropout_rates

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def export_model(self) -> keras.Sequential:
        return self._model
