import tensorflow as tf

from pidgan.algorithms.WGAN_GP import WGAN_GP

LIPSCHITZ_CONSTANT = 1.0


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
        lipschitz_penalty_strategy="two-sided",
        feature_matching_penalty=0.0,
        referee_from_logits=None,
        referee_label_smoothing=None,
        name="CramerGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            lipschitz_penalty=lipschitz_penalty,
            lipschitz_penalty_strategy=lipschitz_penalty_strategy,
            feature_matching_penalty=feature_matching_penalty,
            referee_from_logits=referee_from_logits,
            referee_label_smoothing=referee_label_smoothing,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Energy distance"

        # Critic function
        self._critic = Critic(lambda x, t: self._discriminator(x, training=t))
        if self._bce_loss is None:
            self._referee_critic = Critic(lambda x, t: self._referee(x, training=t))
        else:
            self._referee_critic = None

    def _prepare_trainset(
        self, x, y, sample_weight=None, training_generator=True
    ) -> tuple:
        batch_size = tf.cast(tf.shape(x)[0] / 4, tf.int32)
        x_ref, x_gen_1, x_gen_2 = tf.split(x[: batch_size * 3], 3, axis=0)
        y_ref = y[:batch_size]

        x_gen_concat = tf.concat([x_gen_1, x_gen_2], axis=0)
        y_gen = self._generator(x_gen_concat, training=training_generator)
        y_gen_1, y_gen_2 = tf.split(y_gen, 2, axis=0)

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

    @staticmethod
    def _standard_loss_func(
        critic, trainset_ref, trainset_gen_1, trainset_gen_2, critic_training=False
    ) -> tf.Tensor:
        x_ref, y_ref, w_ref = trainset_ref
        x_gen_1, y_gen_1, w_gen_1 = trainset_gen_1
        x_gen_2, y_gen_2, w_gen_2 = trainset_gen_2

        x_concat_1 = tf.concat([x_ref, x_gen_1], axis=0)
        y_concat_1 = tf.concat([y_ref, y_gen_1], axis=0)
        x_concat_2 = tf.concat([x_gen_2, x_gen_2], axis=0)
        y_concat_2 = tf.concat([y_gen_2, y_gen_2], axis=0)

        d_out = critic(
            (x_concat_1, y_concat_1), (x_concat_2, y_concat_2), training=critic_training
        )
        d_ref, d_gen = tf.split(d_out, 2, axis=0)

        real_loss = tf.reduce_sum(
            w_ref[:, None] * w_gen_2[:, None] * d_ref
        ) / tf.reduce_sum(w_ref * w_gen_2)
        fake_loss = tf.reduce_sum(
            w_gen_1[:, None] * w_gen_2[:, None] * d_gen
        ) / tf.reduce_sum(w_gen_1 * w_gen_2)
        return real_loss - fake_loss

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        return self._standard_loss_func(
            critic=self._critic,
            trainset_ref=trainset_ref,
            trainset_gen_1=trainset_gen_1,
            trainset_gen_2=trainset_gen_2,
            critic_training=False,
        )

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        d_loss = -self._standard_loss_func(
            critic=self._critic,
            trainset_ref=trainset_ref,
            trainset_gen_1=trainset_gen_1,
            trainset_gen_2=trainset_gen_2,
            critic_training=training,
        )
        d_loss += self._lipschitz_regularization(
            self._critic, x, y, sample_weight, training=training
        )
        return d_loss

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        if self._bce_loss is not None:
            return self._bce_loss_func(
                model=self._referee,
                trainset_ref=trainset_ref,
                trainset_gen=trainset_gen_1,
                inj_noise_std=0.0,
                model_training=training,
            )
        else:
            return -self._standard_loss_func(
                critic=self._referee_critic,
                trainset_ref=trainset_ref,
                trainset_gen_1=trainset_gen_1,
                trainset_gen_2=trainset_gen_2,
                critic_training=training,
            )

    def _lipschitz_regularization(
        self, critic, x, y, sample_weight=None, training=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        _, y_ref, _ = trainset_ref
        x_gen_1, y_gen_1, _ = trainset_gen_1
        x_gen_2, y_gen_2, _ = trainset_gen_2

        y_concat = tf.concat([y_ref, y_gen_1], axis=0)

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
                y_gen_1 + eps * (y_ref - y_gen_1),
                clip_value_min=tf.reduce_min(y_concat, axis=0),
                clip_value_max=tf.reduce_max(y_concat, axis=0),
            )
            tape.watch(y_hat)

            # Value of the critic on interpolated points
            c_out_hat = critic((x_gen_1, y_hat), (x_gen_2, y_gen_2), training=training)
            grad = tape.gradient(c_out_hat, y_hat)
            norm = tf.norm(grad, ord=2, axis=-1)

        if self._lipschitz_penalty_strategy == "two-sided":
            gp_term = (norm - LIPSCHITZ_CONSTANT) ** 2
        else:
            gp_term = (tf.maximum(0.0, norm - LIPSCHITZ_CONSTANT)) ** 2
        return self._lipschitz_penalty * tf.reduce_mean(gp_term)
