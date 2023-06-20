import tensorflow as tf

from pidgan.algorithms.WGAN import WGAN

PENALTY_STRATEGIES = ["two-sided", "one-sided"]


class WGAN_GP(WGAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        lipschitz_penalty=1.0,
        penalty_strategy="two-sided",
        from_logits=None,
        label_smoothing=None,
        name="WGAN-GP",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            clip_param=0.01,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            name=name,
            dtype=dtype,
        )
        self._clip_param = None

        # Lipschitz penalty
        assert isinstance(lipschitz_penalty, (int, float))
        assert lipschitz_penalty > 0.0
        self._lipschitz_penalty = float(lipschitz_penalty)

        # Penalty strategy
        assert isinstance(penalty_strategy, str)
        if penalty_strategy not in PENALTY_STRATEGIES:
            raise ValueError(
                "`penalty_strategy` should be selected "
                f"in {PENALTY_STRATEGIES}, instead "
                f"'{penalty_strategy}' passed"
            )
        self._penalty_strategy = penalty_strategy

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
        if self._referee_loss is None:
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
        x_ref, y_ref, _ = trainset_ref
        x_gen, y_gen, _ = trainset_gen

        xy_ref = tf.concat([x_ref, y_ref], axis=1)
        xy_gen = tf.concat([x_gen, y_gen], axis=1)

        with tf.GradientTape() as tape:
            # Compute interpolated points
            eps = tf.random.uniform(
                shape=(tf.shape(xy_ref)[0],), minval=0.0, maxval=1.0, dtype=y_ref.dtype
            )
            xy_hat = xy_gen + tf.tile(eps[:, None], (1, tf.shape(xy_ref)[1])) * (
                xy_ref - xy_gen
            )
            tape.watch(xy_hat)

            # Value of the critic on interpolated points
            x_hat, y_hat = (
                xy_hat[:, : tf.shape(x_ref)[1]],
                xy_hat[:, tf.shape(x_ref)[1] :],
            )
            c_out_hat = critic((x_hat, y_hat), training=training)
            grad = tape.gradient(c_out_hat, xy_hat)
            norm = tf.norm(grad, ord=2, axis=-1)

        if self._penalty_strategy == "two-sided":
            gp_term = (norm - 1.0) ** 2
        else:
            gp_term = (tf.maximum(0.0, norm - 1.0)) ** 2
        return self._lipschitz_penalty * tf.reduce_mean(gp_term)

    @property
    def lipschitz_penalty(self) -> float:
        return self._lipschitz_penalty

    @property
    def penalty_strategy(self) -> str:
        return self._penalty_strategy
