import tensorflow as tf

from pidgan.algorithms.GAN import GAN


class LSGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        minimize_pearson_chi2=False,
        injected_noise_stddev=0,
        name=None,
        dtype=None,
    ) -> None:
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            use_original_loss=True,
            injected_noise_stddev=injected_noise_stddev,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Least squares loss"
        self._use_original_loss = None

        # Flag to minimize the Pearson chi2 divergence
        assert isinstance(minimize_pearson_chi2, bool)
        self._minimize_pearson_chi2 = minimize_pearson_chi2

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        if self._inj_noise_std > 0.0:
            rnd_ref, rnd_gen = tf.split(
                tf.random.normal(
                    shape=(tf.shape(y_ref)[0] * 2, tf.shape(y_ref)[1]),
                    mean=0.0,
                    stddev=self._inj_noise_std,
                    dtype=y_ref.dtype,
                ),
                num_or_size_splits=2,
                axis=0,
            )
        else:
            rnd_ref, rnd_gen = 0.0, 0.0

        d_out_ref = self._discriminator((x_ref, y_ref + rnd_ref), training=False)
        d_out_gen = self._discriminator((x_gen, y_gen + rnd_gen), training=False)

        c = 0.0 if self._minimize_pearson_chi2 else 1.0

        real_loss = tf.reduce_sum(w_ref * (d_out_ref - c) ** 2) / tf.reduce_sum(w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen * (d_out_gen - c) ** 2) / tf.reduce_sum(w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return (real_loss + fake_loss) / 2.0

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        if self._inj_noise_std > 0.0:
            rnd_ref, rnd_gen = tf.split(
                tf.random.normal(
                    shape=(tf.shape(y_ref)[0] * 2, tf.shape(y_ref)[1]),
                    mean=0.0,
                    stddev=self._inj_noise_std,
                    dtype=y_ref.dtype,
                ),
                num_or_size_splits=2,
                axis=0,
            )
        else:
            rnd_ref, rnd_gen = 0.0, 0.0

        d_out_ref = self._discriminator((x_ref, y_ref + rnd_ref), training=training)
        d_out_gen = self._discriminator((x_gen, y_gen + rnd_gen), training=training)

        a = -1.0 if self._minimize_pearson_chi2 else 0.0

        real_loss = tf.reduce_sum(w_ref * (d_out_ref - 1.0) ** 2) / tf.reduce_sum(w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen * (d_out_gen - a) ** 2) / tf.reduce_sum(w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return (real_loss + fake_loss) / 2.0

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        r_out_ref = self._referee((x_ref, y_ref), training=training)
        r_out_gen = self._referee((x_gen, y_gen), training=training)

        a = -1.0 if self._minimize_pearson_chi2 else 0.0

        real_loss = tf.reduce_sum(w_ref * (r_out_ref - 1.0) ** 2) / tf.reduce_sum(w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen * (r_out_gen - a) ** 2) / tf.reduce_sum(w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return (real_loss + fake_loss) / 2.0

    def _compute_threshold(self, model, x, y, sample_weight=None) -> tf.Tensor:
        return 0.0

    @property
    def minimize_pearson_chi2(self) -> bool:
        return self._minimize_pearson_chi2
