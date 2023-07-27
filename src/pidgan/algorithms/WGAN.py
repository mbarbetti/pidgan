import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from pidgan.algorithms.GAN import GAN


class WGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        clip_param=0.01,
        feature_matching_penalty=0.0,
        referee_from_logits=None,
        referee_label_smoothing=None,
        name="WGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            use_original_loss=True,
            injected_noise_stddev=0.0,
            feature_matching_penalty=feature_matching_penalty,
            referee_from_logits=referee_from_logits,
            referee_label_smoothing=referee_label_smoothing,
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

    def _d_train_step(self, x, y, sample_weight=None) -> None:
        super()._d_train_step(x, y, sample_weight)
        for w in self._discriminator.trainable_weights:
            w = tf.clip_by_value(w, -self._clip_param, self._clip_param)

    def _r_train_step(self, x, y, sample_weight=None) -> None:
        super()._r_train_step(x, y, sample_weight)
        if self._bce_loss is None:
            for w in self._referee.trainable_weights:
                w = tf.clip_by_value(w, -self._clip_param, self._clip_param)

    @staticmethod
    def _standard_loss_func(
        model, trainset_ref, trainset_gen, model_training=False
    ) -> tf.Tensor:
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)

        m_out = model((x_concat, y_concat), training=model_training)
        m_ref, m_gen = tf.split(m_out, 2, axis=0)

        real_loss = tf.reduce_sum(w_ref[:, None] * m_ref) / tf.reduce_sum(w_ref)
        fake_loss = tf.reduce_sum(w_gen[:, None] * m_gen) / tf.reduce_sum(w_gen)
        return real_loss - fake_loss

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        return self._standard_loss_func(
            model=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            model_training=False,
        )

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return -self._standard_loss_func(
            model=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            model_training=training,
        )

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        if self._bce_loss is not None:
            return self._bce_loss_func(
                model=self._referee,
                trainset_ref=trainset_ref,
                trainset_gen=trainset_gen,
                inj_noise_std=0.0,
                model_training=training,
            )
        else:
            return -self._standard_loss_func(
                model=self._referee,
                trainset_ref=trainset_ref,
                trainset_gen=trainset_gen,
                model_training=training,
            )

    def _compute_threshold(self, model, x, y, sample_weight=None) -> tf.Tensor:
        return 0.0

    @property
    def clip_param(self) -> float:
        return self._clip_param
