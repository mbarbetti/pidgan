import tensorflow as tf

from pidgan.algorithms.WGAN import WGAN

PENALTY_STRATEGIES = ["two-sided", "one-sided"]


class WGAN_GP(WGAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        lipschitz_penalty=10.0,
        penalty_strategy="two-sided",
        from_logits=None,
        label_smoothing=None,
        name=None,
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
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        d_out_ref = self._discriminator((x_ref, y_ref), training=training)
        d_out_gen = self._discriminator((x_gen, y_gen), training=training)

        real_loss = tf.reduce_sum(w_ref * d_out_ref) / tf.reduce_sum(w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen * d_out_gen) / tf.reduce_sum(w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return (
            fake_loss
            - real_loss
            + self._lipschitz_regularization(
                self._discriminator, x, y, sample_weight, training=training
            )
        )

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        r_out_ref = self._referee((x_ref, y_ref), training=training)
        r_out_gen = self._referee((x_gen, y_gen), training=training)

        if self._referee_loss is not None:
            real_loss = self._referee_loss(
                tf.ones_like(r_out_ref), r_out_ref, sample_weight=w_ref
            )
            real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
            fake_loss = self._referee_loss(
                tf.zeros_like(r_out_gen), r_out_gen, sample_weight=w_gen
            )
            fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
            return (real_loss + fake_loss) / 2.0
        else:
            real_loss = tf.reduce_sum(w_ref * r_out_ref) / tf.reduce_sum(w_ref)
            real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
            fake_loss = tf.reduce_sum(w_gen * r_out_gen) / tf.reduce_sum(w_gen)
            fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
            return (
                fake_loss
                - real_loss
                + self._lipschitz_regularization(
                    self._referee, x, y, sample_weight, training=training
                )
            )

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
