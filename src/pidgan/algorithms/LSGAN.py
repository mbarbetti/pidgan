import tensorflow as tf

from pidgan.algorithms.GAN import GAN


class LSGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        minimize_pearson_chi2=False,
        injected_noise_stddev=0.0,
        feature_matching_penalty=0.0,
        referee=None,
        name="LSGAN",
        dtype=None,
    ) -> None:
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            injected_noise_stddev=injected_noise_stddev,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Least squares loss"
        self._use_original_loss = None

        # Flag to minimize the Pearson chi2 divergence
        assert isinstance(minimize_pearson_chi2, bool)
        self._minimize_pearson_chi2 = minimize_pearson_chi2

    @staticmethod
    def _standard_loss_func(
        discriminator,
        trainset_ref,
        trainset_gen,
        a_param,
        b_param,
        inj_noise_std=0.0,
        training_discriminator=False,
        generator_loss=True,
    ) -> tf.Tensor:
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)

        if inj_noise_std > 0.0:
            rnd_noise = tf.random.normal(
                shape=(tf.shape(y_ref)[0] * 2, tf.shape(y_ref)[1]),
                mean=0.0,
                stddev=inj_noise_std,
                dtype=y_ref.dtype,
            )
            y_concat += rnd_noise

        d_out = discriminator((x_concat, y_concat), training=training_discriminator)
        d_ref, d_gen = tf.split(d_out, 2, axis=0)

        real_loss = tf.reduce_sum(
            w_ref[:, None] * (d_ref - b_param) ** 2
        ) / tf.reduce_sum(w_ref)
        fake_loss = tf.reduce_sum(
            w_gen[:, None] * (d_gen - a_param) ** 2
        ) / tf.reduce_sum(w_gen)

        if generator_loss:
            return (tf.stop_gradient(real_loss) + fake_loss) / 2.0
        else:
            return (real_loss + fake_loss) / 2.0

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        return self._standard_loss_func(
            discriminator=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            a_param=0.0 if self._minimize_pearson_chi2 else 1.0,
            b_param=0.0 if self._minimize_pearson_chi2 else 1.0,
            inj_noise_std=self._inj_noise_std,
            training_discriminator=False,
            generator_loss=True,
        )

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return self._standard_loss_func(
            discriminator=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            a_param=-1.0 if self._minimize_pearson_chi2 else 0.0,
            b_param=1.0,
            inj_noise_std=self._inj_noise_std,
            training_discriminator=training,
            generator_loss=False,
        )

    def _compute_threshold(self, discriminator, x, y, sample_weight=None) -> tf.Tensor:
        return 0.0

    @property
    def minimize_pearson_chi2(self) -> bool:
        return self._minimize_pearson_chi2
