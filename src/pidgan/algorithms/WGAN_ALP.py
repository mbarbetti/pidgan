import tensorflow as tf

from pidgan.algorithms.WGAN_GP import WGAN_GP

LIPSCHITZ_CONSTANT = 1.0
FIXED_XI = 1.0
SAMPLED_XI_MIN = 0.8
SAMPLED_XI_MAX = 1.2
EPSILON = 1e-12


class WGAN_ALP(WGAN_GP):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="one-sided",
        feature_matching_penalty=0.0,
        referee_from_logits=None,
        referee_label_smoothing=None,
        name="WGAN-ALP",
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

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)
        c_out = critic((x_concat, y_concat), training=training)

        # Initial virtual adversarial direction
        adv_dir = tf.random.uniform(
            shape=(2 * tf.shape(y_ref)[0], tf.shape(y_ref)[1]),
            minval=-1.0,
            maxval=1.0,
            dtype=y_ref.dtype,
        )
        adv_dir /= tf.norm(adv_dir, axis=-1, keepdims=True)

        for _ in range(self._vir_dir_upds):
            with tf.GradientTape() as tape:
                tape.watch(adv_dir)
                y_hat = tf.clip_by_value(
                    y_concat + FIXED_XI * adv_dir,
                    clip_value_min=tf.reduce_min(y_concat, axis=0),
                    clip_value_max=tf.reduce_max(y_concat, axis=0),
                )
                c_hat = critic((x_concat, y_hat), training=training)
                c_diff = tf.reduce_mean(tf.abs(c_out - c_hat))
                grad = tape.gradient(c_diff, adv_dir) + EPSILON  # non-zero gradient
                adv_dir = grad / tf.norm(grad, axis=-1, keepdims=True)

        # Virtual adversarial direction
        xi = tf.random.uniform(
            shape=(2 * tf.shape(y_ref)[0],),
            minval=SAMPLED_XI_MIN,
            maxval=SAMPLED_XI_MAX,
            dtype=y_ref.dtype,
        )
        xi = tf.tile(xi[:, None], (1, tf.shape(y_ref)[1]))
        y_hat = tf.clip_by_value(
            y_concat + xi * adv_dir,
            clip_value_min=tf.reduce_min(y_concat, axis=0),
            clip_value_max=tf.reduce_max(y_concat, axis=0),
        )
        c_hat = critic((x_concat, y_hat), training=training)
        c_diff = tf.abs(c_out - c_hat)
        y_diff = tf.norm(
            tf.abs(y_concat - y_hat) + EPSILON,  # non-zero difference
            ord=2,
            axis=-1,
            keepdims=True,
        )

        K = c_diff / y_diff  # lipschitz constant
        if self._lipschitz_penalty_strategy == "two-sided":
            alp_term = tf.abs(K - LIPSCHITZ_CONSTANT)
        else:
            alp_term = tf.maximum(0.0, K - LIPSCHITZ_CONSTANT)
        return self._lipschitz_penalty * tf.reduce_mean(alp_term) ** 2

    @property
    def virtual_direction_upds(self) -> int:
        return self._vir_dir_upds
