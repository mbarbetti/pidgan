import tensorflow as tf

from pidgan.algorithms.WGAN_GP import WGAN_GP

LIPSCHITZ_CONSTANT = 1.0
FIXED_XI = 1.0
SAMPLED_XI_MIN = 0.0
SAMPLED_XI_MAX = 1.0
EPSILON = 1e-12


class WGAN_ALP(WGAN_GP):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        lipschitz_penalty=1.0,
        penalty_strategy="one-sided",
        from_logits=None,
        label_smoothing=None,
        name="WGAN-ALP",
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

    def compile(
        self,
        metrics=None,
        generator_optimizer="rmsprop",
        discriminator_optimizer="rmsprop",
        referee_optimizer="rmsprop",
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        referee_upds_per_batch=1,
        virtual_direction_upds=1,
    ) -> None:
        super().compile(
            metrics=metrics,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            referee_optimizer=referee_optimizer,
            generator_upds_per_batch=generator_upds_per_batch,
            discriminator_upds_per_batch=discriminator_upds_per_batch,
            referee_upds_per_batch=referee_upds_per_batch,
        )

        # Virtual adversarial direction updates
        assert isinstance(virtual_direction_upds, (int, float))
        assert virtual_direction_upds > 0
        self._vir_dir_upds = int(virtual_direction_upds)

    def _lipschitz_regularization(
        self, critic, x, y, sample_weight=None, training=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, _ = trainset_ref
        x_gen, y_gen, _ = trainset_gen

        c_out_ref = critic((x_ref, y_ref), training=training)
        c_out_gen = critic((x_gen, y_gen), training=training)

        xy_ref = tf.concat([x_ref, y_ref], axis=1)
        xy_gen = tf.concat([x_gen, y_gen], axis=1)

        with tf.GradientTape() as tape:
            # Initial virtual adversarial direction
            d = tf.random.uniform(
                shape=(2 * tf.shape(xy_ref)[0], tf.shape(xy_ref)[1]),
                minval=-1.0,
                maxval=1.0,
                dtype=y_ref.dtype,
            )
            d /= tf.norm(d, ord=2, axis=-1)[:, None]
            tape.watch(d)

            for _ in range(self._vir_dir_upds):
                d_ref, d_gen = tf.split(d, 2, axis=0)
                xy_ref_hat = tf.clip_by_value(
                    xy_ref + FIXED_XI * d_ref,
                    clip_value_min=tf.reduce_min(xy_ref, axis=0),
                    clip_value_max=tf.reduce_max(xy_ref, axis=0),
                )
                xy_gen_hat = tf.clip_by_value(
                    xy_gen + FIXED_XI * d_gen,
                    clip_value_min=tf.reduce_min(xy_gen, axis=0),
                    clip_value_max=tf.reduce_max(xy_gen, axis=0),
                )
                x_ref_hat, y_ref_hat = (
                    xy_ref_hat[:, : tf.shape(x_ref)[1]],
                    xy_ref_hat[:, tf.shape(x_ref)[1] :],
                )
                c_out_ref_hat = critic((x_ref_hat, y_ref_hat), training=training)
                x_gen_hat, y_gen_hat = (
                    xy_gen_hat[:, : tf.shape(x_ref)[1]],
                    xy_gen_hat[:, tf.shape(x_ref)[1] :],
                )
                c_out_gen_hat = critic((x_gen_hat, y_gen_hat), training=training)

                c_out_diff = tf.reduce_mean(
                    tf.abs(
                        tf.concat([c_out_ref, c_out_gen], axis=0)
                        - tf.concat([c_out_ref_hat, c_out_gen_hat], axis=0)
                    )
                )
                grad = tape.gradient(c_out_diff, d) + EPSILON  # non-zero gradient
                d = grad / tf.norm(grad, ord=2, axis=-1, keepdims=True)

        # Virtual adversarial direction
        xi = tf.random.uniform(
            shape=(2 * tf.shape(xy_ref)[0],),
            minval=SAMPLED_XI_MIN,
            maxval=SAMPLED_XI_MAX,
            dtype=y_ref.dtype,
        )
        xi = tf.tile(xi[:, None], (1, tf.shape(xy_ref)[1]))
        xi_ref, xi_gen = tf.split(xi, 2, axis=0)
        d_ref, d_gen = tf.split(d, 2, axis=0)
        xy_ref_hat = tf.clip_by_value(
            xy_ref + (0.5 + xi_ref) * d_ref,
            clip_value_min=tf.reduce_min(xy_ref, axis=0),
            clip_value_max=tf.reduce_max(xy_ref, axis=0),
        )
        xy_gen_hat = tf.clip_by_value(
            xy_gen + (0.5 + xi_gen) * d_gen,
            clip_value_min=tf.reduce_min(xy_gen, axis=0),
            clip_value_max=tf.reduce_max(xy_gen, axis=0),
        )

        x_ref_hat, y_ref_hat = (
            xy_ref_hat[:, : tf.shape(x_ref)[1]],
            xy_ref_hat[:, tf.shape(x_ref)[1] :],
        )
        c_out_ref_hat = critic((x_ref_hat, y_ref_hat), training=training)
        x_gen_hat, y_gen_hat = (
            xy_gen_hat[:, : tf.shape(x_ref)[1]],
            xy_gen_hat[:, tf.shape(x_ref)[1] :],
        )
        c_out_gen_hat = critic((x_gen_hat, y_gen_hat), training=training)

        c_out_diff = tf.reduce_mean(
            tf.abs(
                tf.concat([c_out_ref, c_out_gen], axis=0)
                - tf.concat([c_out_ref_hat, c_out_gen_hat], axis=0)
            )
        )
        xy_diff = tf.norm(
            tf.abs(
                tf.concat([xy_ref, xy_gen], axis=0)
                - tf.concat([xy_ref_hat, xy_gen_hat], axis=0)
            )
            + EPSILON,  # non-zero difference
            ord=2,
            axis=-1,
            keepdims=True,
        )

        K = c_out_diff / xy_diff  # lipschitz constant
        if self._penalty_strategy == "two-sided":
            alp_term = tf.abs(K - LIPSCHITZ_CONSTANT)
        else:
            alp_term = tf.maximum(0.0, K - LIPSCHITZ_CONSTANT)
        return self._lipschitz_penalty * tf.reduce_mean(alp_term) ** 2

    @property
    def virtual_direction_upds(self) -> int:
        return self._vir_dir_upds
