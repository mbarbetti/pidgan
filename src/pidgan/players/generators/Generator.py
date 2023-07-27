import tensorflow as tf
from tensorflow import keras

LEAKY_ALPHA = 0.1
SEED = 42


class Generator(keras.Model):
    def __init__(
        self,
        output_dim,
        latent_dim,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        dropout_rate=0.0,
        output_activation=None,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)

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

    def call(self, x) -> tf.Tensor:
        x, _ = self._prepare_input(x, seed=None)
        out = self._seq(x)
        return out

    def summary(self, **kwargs) -> None:
        self._seq.summary(**kwargs)

    def generate(self, x, seed=None, return_latent_sample=False) -> tf.Tensor:
        tf.random.set_seed(seed=SEED)
        x, latent_sample = self._prepare_input(x, seed=seed)
        out = self._seq(x)
        if return_latent_sample:
            return out, latent_sample
        else:
            return out

    def _prepare_input(self, x, seed=None) -> tuple:
        latent_sample = tf.random.normal(
            shape=(tf.shape(x)[0], self._latent_dim),
            mean=0.0,
            stddev=1.0,
            dtype=self.dtype,
            seed=seed,
        )
        x = tf.concat([x, latent_sample], axis=-1)
        return x, latent_sample

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

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
