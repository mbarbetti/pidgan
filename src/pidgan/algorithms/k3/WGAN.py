import keras as k

from pidgan.algorithms.k3.GAN import GAN


class WGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        clip_param=0.01,
        feature_matching_penalty=0.0,
        referee=None,
        name="WGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Wasserstein distance"
        self._use_original_loss = None
        self._inj_noise_std = None

        # Clipping parameter
        assert isinstance(clip_param, (int, float))
        assert clip_param > 0.0
        self._clip_param = float(clip_param)

    def _tf_d_train_step(self, x, y, sample_weight=None) -> None:
        super()._tf_d_train_step(x, y, sample_weight)
        for w in self._discriminator.trainable_weights:
            w = k.ops.clip(w, -self._clip_param, self._clip_param)

    @staticmethod
    def _standard_loss_func(
        discriminator,
        trainset_ref,
        trainset_gen,
        training_discriminator=False,
        generator_loss=True,
    ):
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        x_concat = k.ops.concatenate([x_ref, x_gen], axis=0)
        y_concat = k.ops.concatenate([y_ref, y_gen], axis=0)

        d_out = discriminator((x_concat, y_concat), training=training_discriminator)
        d_ref, d_gen = k.ops.split(d_out, 2, axis=0)

        real_loss = k.ops.sum(w_ref[:, None] * d_ref) / k.ops.sum(w_ref)
        fake_loss = k.ops.sum(w_gen[:, None] * d_gen) / k.ops.sum(w_gen)

        if generator_loss:
            return k.ops.stop_gradient(real_loss) - fake_loss
        else:
            return fake_loss - real_loss

    def _compute_g_loss(self, x, y, sample_weight=None, training=True, test=False):
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        return self._standard_loss_func(
            discriminator=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            training_discriminator=False,
            generator_loss=True,
        )

    def _compute_d_loss(self, x, y, sample_weight=None, training=True, test=False):
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return self._standard_loss_func(
            discriminator=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            training_discriminator=training,
            generator_loss=False,
        )

    def _compute_threshold(self, discriminator, x, y, sample_weight=None):
        return 0.0

    @property
    def clip_param(self) -> float:
        return self._clip_param
