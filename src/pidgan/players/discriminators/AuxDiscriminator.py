import tensorflow as tf

from pidgan.players.discriminators.Discriminator import Discriminator


class AuxDiscriminator(Discriminator):
    def __init__(
        self,
        output_dim,
        aux_features,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        dropout_rate=0,
        output_activation="sigmoid",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_units=mlp_hidden_units,
            dropout_rate=dropout_rate,
            output_activation=output_activation,
            name=name,
            dtype=dtype,
        )

        self._aux_features = list()
        if isinstance(aux_features, str):
            aux_features = [aux_features]

        self._aux_indices = list()
        self._aux_operators = list()
        for aux_feat in aux_features:
            assert isinstance(aux_feat, str)
            if "+" in aux_feat:
                self._aux_operators.append(tf.math.add)
                self._aux_indices.append([int(i) for i in aux_feat.split("+")])
            elif "-" in aux_feat:
                self._aux_operators.append(tf.math.subtract)
                self._aux_indices.append([int(i) for i in aux_feat.split("-")])
            elif "*" in aux_feat:
                self._aux_operators.append(tf.math.multiply)
                self._aux_indices.append([int(i) for i in aux_feat.split("*")])
            elif "/" in aux_feat:
                self._aux_operators.append(tf.math.divide)
                self._aux_indices.append([int(i) for i in aux_feat.split("/")])
            else:
                raise ValueError("")
            self._aux_features.append(aux_feat)

    def _prepare_input(self, inputs) -> tf.Tensor:
        x, y = inputs
        new_inputs = [x, y]
        for aux_idx, aux_op in zip(self._aux_indices, self._aux_operators):
            new_inputs.append(aux_op(y[:, aux_idx[0]], y[:, aux_idx[1]])[:, None])
        d_in = tf.concat(new_inputs, axis=1)
        return d_in

    @property
    def aux_features(self) -> list:
        return self._aux_features
