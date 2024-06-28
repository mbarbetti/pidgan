import keras as k
from pidgan.players.discriminators.k3.ResDiscriminator import ResDiscriminator


class AuxDiscriminator(ResDiscriminator):
    def __init__(
        self,
        output_dim,
        aux_features,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0,
        enable_residual_blocks=False,
        output_activation="sigmoid",
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            mlp_hidden_units=mlp_hidden_units,
            mlp_dropout_rates=mlp_dropout_rates,
            output_activation=output_activation,
            name=name,
            dtype=dtype,
        )

        # Residual blocks
        assert isinstance(enable_residual_blocks, bool)
        self._enable_res_blocks = enable_residual_blocks

        # Auxiliary features
        self._aux_features = list()
        if isinstance(aux_features, str):
            aux_features = [aux_features]

        self._aux_indices = list()
        self._aux_operators = list()
        for aux_feat in aux_features:
            assert isinstance(aux_feat, str)
            if "+" in aux_feat:
                self._aux_operators.append(k.ops.add)
                self._aux_indices.append([int(i) for i in aux_feat.split("+")])
            elif "-" in aux_feat:
                self._aux_operators.append(k.ops.subtract)
                self._aux_indices.append([int(i) for i in aux_feat.split("-")])
            elif "*" in aux_feat:
                self._aux_operators.append(k.ops.multiply)
                self._aux_indices.append([int(i) for i in aux_feat.split("*")])
            elif "/" in aux_feat:
                self._aux_operators.append(k.ops.divide)
                self._aux_indices.append([int(i) for i in aux_feat.split("/")])
            else:
                raise ValueError(
                    f"Operator for auxiliary features not supported. "
                    f"Operators should be selected in ['+', '-', '*', '/'], "
                    f"instead '{aux_feat}' passed."
                )
            self._aux_features.append(aux_feat)

    def _get_input_dim(self, input_shape) -> int:
        in_dim = super()._get_input_dim(input_shape)
        in_dim += len(self._aux_features)
        return in_dim

    def _prepare_input(self, x):
        in_feats = super()._prepare_input(x)
        if isinstance(x, (list, tuple)):
            _, y = x
        else:
            y = x
        aux_feats = list()
        for aux_idx, aux_op in zip(self._aux_indices, self._aux_operators):
            aux_feats.append(aux_op(y[:, aux_idx[0]], y[:, aux_idx[1]])[:, None])
        self._aux_feats = k.ops.concatenate(aux_feats, axis=-1)
        return k.ops.concatenate([in_feats, self._aux_feats], axis=-1)

    def call(self, x, return_aux_features=False):
        out = super().call(x)
        if return_aux_features:
            return out, self._aux_feats
        else:
            return out

    @property
    def aux_features(self) -> list:
        return self._aux_features

    @property
    def enable_residual_blocks(self) -> bool:
        return self._enable_res_blocks
