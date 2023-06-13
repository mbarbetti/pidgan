import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy as TF_BCE

from pidgan.algorithms.GAN import GAN


class BceGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        injected_noise_stddev=0.0,
        from_logits=False,
        label_smoothing=0.0,
        name=None,
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            use_original_loss=True,
            injected_noise_stddev=injected_noise_stddev,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Binary cross-entropy"
        self._use_original_loss = None

        # TensorFlow BinaryCrossentropy
        self._loss = TF_BCE(from_logits=from_logits, label_smoothing=label_smoothing)
        self._from_logits = bool(from_logits)
        self._label_smoothing = float(label_smoothing)

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        _, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        x_gen, y_gen, w_gen = trainset_gen

        if self._inj_noise_std > 0.0:
            rnd_gen = tf.random.normal(
                shape=tf.shape(y_gen),
                mean=0.0,
                stddev=self._inj_noise_std,
                dtype=y_gen.dtype,
            )
        else:
            rnd_gen = 0.0

        d_out_gen = self._discriminator((x_gen, y_gen + rnd_gen), training=False)

        fake_loss = self._loss(tf.ones_like(d_out_gen), d_out_gen, sample_weight=w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_gen.dtype)
        return fake_loss

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

        real_loss = self._loss(tf.ones_like(d_out_ref), d_out_ref, sample_weight=w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)

        fake_loss = self._loss(tf.zeros_like(d_out_gen), d_out_gen, sample_weight=w_gen)
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

        real_loss = self._loss(tf.ones_like(r_out_ref), r_out_ref, sample_weight=w_ref)
        real_loss = tf.cast(real_loss, dtype=y_ref.dtype)
        fake_loss = self._loss(tf.zeros_like(r_out_gen), r_out_gen, sample_weight=w_gen)
        fake_loss = tf.cast(fake_loss, dtype=y_ref.dtype)
        return (real_loss + fake_loss) / 2.0

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
