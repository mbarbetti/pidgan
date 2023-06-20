import tensorflow as tf

from pidgan.algorithms.WGAN_GP import WGAN_GP


class Critic:
    def __init__(self, h) -> None:
        self._h = h

    def __call__(self, input_1, input_2, training=True) -> tf.Tensor:
        metric = tf.norm(
            self._h(input_1, training) - self._h(input_2, training), axis=-1
        ) - tf.norm(self._h(input_1, training), axis=-1)
        return metric


class CramerGAN(WGAN_GP):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        lipschitz_penalty=1.0,
        penalty_strategy="two-sided",
        from_logits=None,
        label_smoothing=None,
        name="CramerGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            lipschitz_penalty=lipschitz_penalty,
            penalty_strategy=penalty_strategy,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Energy distance"

        # Critic function
        self._critic = Critic(lambda x, t: self._discriminator(x, training=t))
        if self._referee_loss is None:
            self._referee_critic = Critic(lambda x, t: self._referee(x, training=t))
        else:
            self._referee_critic = None

    def _prepare_trainset(
        self, x, y, sample_weight=None, training_generator=True
    ) -> tuple:
        batch_size = tf.cast(tf.shape(x)[0] / 4, tf.int32)
        x_ref, x_gen_1, x_gen_2 = tf.split(x[: batch_size * 3], 3, axis=0)
        y_ref = y[:batch_size]
        y_gen_1 = self._generator(x_gen_1, training=training_generator)
        y_gen_2 = self._generator(x_gen_2, training=training_generator)

        if sample_weight is not None:
            w_ref, w_gen_1, w_gen_2 = tf.split(
                sample_weight[: batch_size * 3], 3, axis=0
            )
        else:
            w_ref, w_gen_1, w_gen_2 = tf.split(
                tf.ones(shape=(batch_size * 3,)), 3, axis=0
            )

        return (
            (x_ref, y_ref, w_ref),
            (x_gen_1, y_gen_1, w_gen_1),
            (x_gen_2, y_gen_2, w_gen_2),
        )

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen_1, y_gen_1, w_gen_1 = trainset_gen_1
        x_gen_2, y_gen_2, w_gen_2 = trainset_gen_2

        d_out_ref = self._critic((x_ref, y_ref), (x_gen_2, y_gen_2), training=False)
        d_out_gen = self._critic((x_gen_1, y_gen_1), (x_gen_2, y_gen_2), training=False)

        real_loss = tf.reduce_sum(w_ref * w_gen_2 * d_out_ref) / tf.reduce_sum(
            w_ref * w_gen_2
        )
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen_1 * w_gen_2 * d_out_gen) / tf.reduce_sum(
            w_gen_1 * w_gen_2
        )
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return real_loss - fake_loss

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen_1, y_gen_1, w_gen_1 = trainset_gen_1
        x_gen_2, y_gen_2, w_gen_2 = trainset_gen_2

        d_out_ref = self._critic((x_ref, y_ref), (x_gen_2, y_gen_2), training=training)
        d_out_gen = self._critic(
            (x_gen_1, y_gen_1), (x_gen_2, y_gen_2), training=training
        )

        real_loss = tf.reduce_sum(w_ref * w_gen_2 * d_out_ref) / tf.reduce_sum(
            w_ref * w_gen_2
        )
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen_1 * w_gen_2 * d_out_gen) / tf.reduce_sum(
            w_gen_1 * w_gen_2
        )
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return (
            fake_loss
            - real_loss
            + self._lipschitz_regularization(
                self._critic, x, y, sample_weight, training=training
            )
        )

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen_1, y_gen_1, w_gen_1 = trainset_gen_1
        x_gen_2, y_gen_2, w_gen_2 = trainset_gen_2

        if self._referee_loss is not None:
            r_out_ref = self._referee((x_ref, y_ref), training=training)
            r_out_gen = self._referee((x_gen_1, y_gen_1), training=training)

            real_loss = self._referee_loss(
                tf.ones_like(r_out_ref), r_out_ref, sample_weight=w_ref
            )
            real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
            fake_loss = self._referee_loss(
                tf.zeros_like(r_out_gen), r_out_gen, sample_weight=w_gen_1
            )
            fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
            return (real_loss + fake_loss) / 2.0
        else:
            r_out_ref = self._referee_critic(
                (x_ref, y_ref), (x_gen_2, y_gen_2), training=training
            )
            r_out_gen = self._referee_critic(
                (x_gen_1, y_gen_1), (x_gen_2, y_gen_2), training=training
            )

            real_loss = tf.reduce_sum(w_ref * w_gen_2 * r_out_ref) / tf.reduce_sum(
                w_ref * w_gen_2
            )
            real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
            fake_loss = tf.reduce_sum(w_gen_1 * w_gen_2 * r_out_gen) / tf.reduce_sum(
                w_gen_1 * w_gen_2
            )
            fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
            return (
                fake_loss
                - real_loss
                + self._lipschitz_regularization(
                    self._referee_critic, x, y, sample_weight, training=training
                )
            )

    def _lipschitz_regularization(
        self, critic, x, y, sample_weight=None, training=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, _ = trainset_ref
        x_gen_1, y_gen_1, _ = trainset_gen_1
        x_gen_2, y_gen_2, _ = trainset_gen_2

        xy_ref = tf.concat([x_ref, y_ref], axis=1)
        xy_gen_1 = tf.concat([x_gen_1, y_gen_1], axis=1)

        with tf.GradientTape() as tape:
            # Compute interpolated points
            eps = tf.random.uniform(
                shape=(tf.shape(xy_ref)[0],), minval=0.0, maxval=1.0, dtype=y_ref.dtype
            )
            xy_hat = xy_gen_1 + tf.tile(eps[:, None], (1, tf.shape(xy_ref)[1])) * (
                xy_ref - xy_gen_1
            )
            tape.watch(xy_hat)

            # Value of the critic on interpolated points
            x_hat, y_hat = (
                xy_hat[:, : tf.shape(x_ref)[1]],
                xy_hat[:, tf.shape(x_ref)[1] :],
            )
            c_out_hat = critic((x_hat, y_hat), (x_gen_2, y_gen_2), training=training)
            grad = tape.gradient(c_out_hat, xy_hat)
            norm = tf.norm(grad, ord=2, axis=-1)

        if self._penalty_strategy == "two-sided":
            gp_term = (norm - 1.0) ** 2
        else:
            gp_term = (tf.maximum(0.0, norm - 1.0)) ** 2
        return self._lipschitz_penalty * tf.reduce_mean(gp_term)
