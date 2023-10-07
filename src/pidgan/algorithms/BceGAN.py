import tensorflow as tf
from tensorflow import keras

from pidgan.algorithms.GAN import GAN


class BceGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        from_logits=False,
        label_smoothing=0.0,
        injected_noise_stddev=0.0,
        feature_matching_penalty=0.0,
        referee=None,
        name="BceGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            injected_noise_stddev=injected_noise_stddev,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Binary cross-entropy"
        self._use_original_loss = None

        # Keras BinaryCrossentropy
        self._bce_loss = keras.losses.BinaryCrossentropy(
            from_logits=from_logits, label_smoothing=label_smoothing
        )
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
            y_gen += rnd_gen

        d_out_gen = self._discriminator((x_gen, y_gen), training=False)

        fake_loss = self._bce_loss(
            tf.ones_like(d_out_gen), d_out_gen, sample_weight=w_gen
        )
        return fake_loss

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)

        if self._inj_noise_std > 0.0:
            rnd_noise = tf.random.normal(
                shape=tf.shape(y_concat),
                mean=0.0,
                stddev=self._inj_noise_std,
                dtype=y_concat.dtype,
            )
            y_concat += rnd_noise

        d_out = self._discriminator((x_concat, y_concat), training=training)
        d_ref, d_gen = tf.split(d_out, 2, axis=0)

        real_loss = self._bce_loss(tf.ones_like(d_ref), d_ref, sample_weight=w_ref)
        fake_loss = self._bce_loss(tf.zeros_like(d_gen), d_gen, sample_weight=w_gen)
        return (real_loss + fake_loss) / 2.0

    def _compute_threshold(self, model, x, y, sample_weight=None) -> tf.Tensor:
        return 0.0

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
