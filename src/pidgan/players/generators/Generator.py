import tensorflow as tf
from tensorflow import keras

LEAKY_ALPHA = 0.1


class Generator(keras.Model):
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
        super().__init__(name=name, dtype=dtype)
        self._hidden_activation_func = None
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

    def _define_arch(self) -> keras.Sequential:
        model = keras.Sequential(name=f"{self.name}_seq" if self.name else None)
        for i, (units, rate) in enumerate(
            zip(self._mlp_hidden_units, self._mlp_dropout_rates)
        ):
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=self._hidden_activation_func,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if self.name else None,
                    dtype=self.dtype,
                )
            )
            if self._hidden_activation_func is None:
                model.add(
                    keras.layers.LeakyReLU(
                        alpha=LEAKY_ALPHA, name=f"leaky_relu_{i}" if self.name else None
                    )
                )
            model.add(
                keras.layers.Dropout(
                    rate=rate, name=f"dropout_{i}" if self.name else None
                )
            )
        model.add(
            keras.layers.Dense(
                units=self._output_dim,
                activation=self._output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="dense_out" if self.name else None,
                dtype=self.dtype,
            )
        )
        return model

    def _build_model(self, x) -> None:
        if self._model is None:
            self._model = self._define_arch()
        else:
            pass

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

    def call(self, x) -> tf.Tensor:
        x, _ = self._prepare_input(x, seed=None)
        self._build_model(x)
        out = self._model(x)
        return out

    def summary(self, **kwargs) -> None:
        self._model.summary(**kwargs)

    def generate(self, x, seed=None, return_latent_sample=False) -> tf.Tensor:
        tf.random.set_seed(seed=seed)
        x, latent_sample = self._prepare_input(x, seed=seed)
        self._build_model(x)
        out = self._model(x)
        if return_latent_sample:
            return out, latent_sample
        else:
            return out

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
    def mlp_dropout_rates(self) -> list:
        return self._mlp_dropout_rates

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def export_model(self) -> keras.Sequential:
        return self._model
