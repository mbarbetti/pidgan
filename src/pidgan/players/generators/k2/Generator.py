import keras as k
import tensorflow as tf

LEAKY_ALPHA = 0.1


class Generator(k.Model):
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

    def _get_input_dim(self, input_shape) -> int:
        return input_shape[-1] + self._latent_dim

    def build(self, input_shape) -> None:
        in_dim = self._get_input_dim(input_shape)
        seq = k.Sequential(name=f"{self.name}_seq" if self.name else None)
        seq.add(k.layers.InputLayer(input_shape=(in_dim,)))
        for i, (units, rate) in enumerate(
            zip(self._mlp_hidden_units, self._mlp_dropout_rates)
        ):
            seq.add(
                k.layers.Dense(
                    units=units,
                    activation=self._hidden_activation_func,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}" if self.name else None,
                    dtype=self.dtype,
                )
            )
            if self._hidden_activation_func is None:
                seq.add(
                    k.layers.LeakyReLU(
                        alpha=LEAKY_ALPHA, name=f"leaky_relu_{i}" if self.name else None
                    )
                )
            seq.add(
                k.layers.Dropout(rate=rate, name=f"dropout_{i}" if self.name else None)
            )
        seq.add(
            k.layers.Dense(
                units=self._output_dim,
                activation=self._output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="dense_out" if self.name else None,
                dtype=self.dtype,
            )
        )
        self._model = seq

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
        # TODO: add warning for model.build()
        x, _ = self._prepare_input(x, seed=None)
        out = self._model(x)
        return out

    def summary(self, **kwargs) -> None:
        self._model.summary(**kwargs)

    def generate(self, x, seed=None, return_latent_sample=False) -> tf.Tensor:
        # TODO: add warning for model.build()
        tf.random.set_seed(seed=seed)
        x, latent_sample = self._prepare_input(x, seed=seed)
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
    def export_model(self) -> k.Sequential:
        return self._model
