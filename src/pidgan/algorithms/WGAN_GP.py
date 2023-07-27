import tensorflow as tf

from pidgan.algorithms.WGAN import WGAN

PENALTY_STRATEGIES = ["two-sided", "one-sided"]
LIPSCHITZ_CONSTANT = 1.0


class WGAN_GP(WGAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="two-sided",
        feature_matching_penalty=0.0,
        referee_from_logits=None,
        referee_label_smoothing=None,
        name="WGAN-GP",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            clip_param=0.01,
            feature_matching_penalty=feature_matching_penalty,
            referee_from_logits=referee_from_logits,
            referee_label_smoothing=referee_label_smoothing,
            name=name,
            dtype=dtype,
        )
        self._clip_param = None

        # Lipschitz penalty
        assert isinstance(lipschitz_penalty, (int, float))
        assert lipschitz_penalty > 0.0
        self._lipschitz_penalty = float(lipschitz_penalty)

        # Penalty strategy
        assert isinstance(lipschitz_penalty_strategy, str)
        if lipschitz_penalty_strategy not in PENALTY_STRATEGIES:
            raise ValueError(
                "`lipschitz_penalty_strategy` should be selected "
                f"in {PENALTY_STRATEGIES}, instead "
                f"'{lipschitz_penalty_strategy}' passed"
            )
        self._lipschitz_penalty_strategy = lipschitz_penalty_strategy

    def _d_train_step(self, x, y, sample_weight=None) -> None:
        super(WGAN, self)._d_train_step(x, y, sample_weight)

    def _r_train_step(self, x, y, sample_weight=None) -> None:
        super(WGAN, self)._r_train_step(x, y, sample_weight)

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        d_loss = super()._compute_d_loss(x, y, sample_weight, training)
        d_loss += self._lipschitz_regularization(
            self._discriminator, x, y, sample_weight, training=training
        )
        return d_loss

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        r_loss = super()._compute_r_loss(x, y, sample_weight, training)
        if self._bce_loss is None:
            r_loss += self._lipschitz_regularization(
                self._referee, x, y, sample_weight, training=training
            )
        return r_loss

    def _lipschitz_regularization(
        self, critic, x, y, sample_weight=None, training=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        _, y_ref, _ = trainset_ref
        x_gen, y_gen, _ = trainset_gen

        y_concat = tf.concat([y_ref, y_gen], axis=0)

        with tf.GradientTape() as tape:
            # Compute interpolated points
            eps = tf.tile(
                tf.random.uniform(
                    shape=(tf.shape(y_ref)[0],),
                    minval=0.0,
                    maxval=1.0,
                    dtype=y_ref.dtype,
                )[:, None],
                (1, tf.shape(y_ref)[1]),
            )
            y_hat = tf.clip_by_value(
                y_gen + eps * (y_ref - y_gen),
                clip_value_min=tf.reduce_min(y_concat, axis=0),
                clip_value_max=tf.reduce_max(y_concat, axis=0),
            )
            tape.watch(y_hat)

            # Value of the critic on interpolated points
            c_out_hat = critic((x_gen, y_hat), training=training)
            grad = tape.gradient(c_out_hat, y_hat)
            norm = tf.norm(grad, axis=-1)

        if self._lipschitz_penalty_strategy == "two-sided":
            gp_term = (norm - LIPSCHITZ_CONSTANT) ** 2
        else:
            gp_term = (tf.maximum(0.0, norm - LIPSCHITZ_CONSTANT)) ** 2
        return self._lipschitz_penalty * tf.reduce_mean(gp_term)

    @property
    def lipschitz_penalty(self) -> float:
        return self._lipschitz_penalty

    @property
    def lipschitz_penalty_strategy(self) -> str:
        return self._lipschitz_penalty_strategy
