import tensorflow as tf
from tensorflow import keras

from pidgan.algorithms.GAN import GAN


class BceGAN(GAN):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        injected_noise_stddev=0.0,
        feature_matching_penalty=0.0,
        from_logits=False,
        label_smoothing=0.0,
        name="BceGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            referee=referee,
            use_original_loss=True,
            injected_noise_stddev=injected_noise_stddev,
            feature_matching_penalty=feature_matching_penalty,
            referee_from_logits=None,
            referee_label_smoothing=None,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Binary cross-entropy"
        self._use_original_loss = None

        # TensorFlow BinaryCrossentropy
        self._bce_loss = keras.losses.BinaryCrossentropy(
            from_logits=from_logits, label_smoothing=label_smoothing
        )
        self._from_logits = bool(from_logits)
        self._label_smoothing = float(label_smoothing)

        self._referee_from_logits = self._from_logits
        self._referee_label_smoothing = self._label_smoothing

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

        fake_loss = self._bce_loss(
            tf.ones_like(d_out_gen), d_out_gen, sample_weight=w_gen
        )
        return fake_loss

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return self._bce_loss_func(
            model=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            inj_noise_std=self._inj_noise_std,
            model_training=training,
        )

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return self._bce_loss_func(
            model=self._referee,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            inj_noise_std=0.0,
            model_training=training,
        )

    def _compute_threshold(self, model, x, y, sample_weight=None) -> tf.Tensor:
        return 0.0

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
