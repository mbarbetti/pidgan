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
        from_logits=None,
        label_smoothing=None,
        name=None,
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            use_original_loss=True,
            injected_noise_stddev=0.0,
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

        # TensorFlow BinaryCrossentropy
        if (from_logits is not None) and (label_smoothing is not None):
            self._referee_loss = TF_BCE(
                from_logits=from_logits, label_smoothing=label_smoothing
            )
            self._from_logits = bool(from_logits)
            self._label_smoothing = float(label_smoothing)
            self._referee_loss_name = "Binary cross-entropy"
        else:
            self._referee_loss = None
            self._from_logits = None
            self._label_smoothing = None
            self._referee_loss_name = f"{self._loss_name}"

    def _d_train_step(self, x, y, sample_weight=None) -> None:
        super()._d_train_step(x, y, sample_weight)
        for w in self._discriminator.trainable_weights:
            w = tf.clip_by_value(w, -self._clip_param, self._clip_param)

    def _r_train_step(self, x, y, sample_weight=None) -> None:
        super()._r_train_step(x, y, sample_weight)
        if self._referee_loss is None:
            for w in self._referee.trainable_weights:
                w = tf.clip_by_value(w, -self._clip_param, self._clip_param)

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        d_out_ref = self._discriminator((x_ref, y_ref), training=False)
        d_out_gen = self._discriminator((x_gen, y_gen), training=False)

        real_loss = tf.reduce_sum(w_ref * d_out_ref) / tf.reduce_sum(w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = tf.reduce_sum(w_gen * d_out_gen) / tf.reduce_sum(w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return real_loss - fake_loss

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
        return fake_loss - real_loss

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
            return fake_loss - real_loss

    @property
    def clip_param(self) -> float:
        return self._clip_param

    @property
    def from_logits(self):  # TODO: add Union[None, bool]
        return self._from_logits

    @property
    def label_smoothing(self):  # TODO: add Union[None, float]
        return self._label_smoothing
