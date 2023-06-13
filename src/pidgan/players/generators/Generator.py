import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU

LEAKY_ALPHA = 0.1


class Generator(tf.keras.Model):
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
        self._model = tf.keras.Sequential()
        for i, (units, rate) in enumerate(zip(self._mlp_hidden_units, self._dropout_rate)):
            self._model.add(
                Dense(
                    units=units,
                    activation=None,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"gen_dense_{i}" if name else None,
                    dtype=self.dtype,
                )
            )
            self._model.add(LeakyReLU(alpha=LEAKY_ALPHA))
            self._model.add(Dropout(rate=rate))
        self._model.add(
            Dense(
                units=output_dim,
                activation=output_activation,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name=f"gen_dense_out" if name else None,
                dtype=self.dtype,
            )
        )
    
    def call(self, x) -> tf.Tensor:
        x = self._prepare_input(x, seed=None)
        out = self._model(x)
        tf.print("shape:", tf.shape(out))
        return out

    def generate(self, x, seed=None) -> tf.Tensor:
        x = self._prepare_input(x, seed=seed)
        out = self._model(x)
        return out
    
    def _prepare_input(self, x, seed=None) -> tf.Tensor:
        latent_sample = tf.random.normal(
            shape=(tf.shape(x)[0], self._latent_dim),
            mean=0.0,
            stddev=1.0,
            dtype=self.dtype,
            seed=seed,
        )
        x = tf.concat([x, latent_sample], axis=1)
        return x

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
    def model(self) -> tf.keras.Model:
        return self._model
