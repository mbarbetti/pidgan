import tensorflow as tf

from pidgan.algorithms.lipschitz_regularizations import compute_CriticGradientPenalty
from pidgan.algorithms.WGAN_GP import WGAN_GP

LIPSCHITZ_CONSTANT = 1.0


class Critic:
    def __init__(self, nn) -> None:
        self._nn = nn

    def __call__(self, input_1, input_2, training=True) -> tf.Tensor:
        return tf.norm(
            self._nn(input_1, training) - self._nn(input_2, training), axis=-1
        ) - tf.norm(self._nn(input_1, training), axis=-1)


class CramerGAN(WGAN_GP):
    def __init__(
        self,
        generator,
        discriminator,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="two-sided",
        feature_matching_penalty=0.0,
        referee=None,
        name="CramerGAN",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=lipschitz_penalty,
            lipschitz_penalty_strategy=lipschitz_penalty_strategy,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )
        self._loss_name = "Energy distance"

        # Critic function
        self._critic = Critic(lambda x, t: self._discriminator(x, training=t))

    def _prepare_trainset(
        self, x, y, sample_weight=None, training_generator=True
    ) -> tuple:
        batch_size = tf.cast(tf.shape(x)[0] / 4, tf.int32)
        x_ref, x_gen_1, x_gen_2 = tf.split(x[: batch_size * 3], 3, axis=0)
        y_ref = y[:batch_size]

        x_gen_concat = tf.concat([x_gen_1, x_gen_2], axis=0)
        y_gen = self._generator(x_gen_concat, training=training_generator)
        y_gen_1, y_gen_2 = tf.split(y_gen, 2, axis=0)

        if sample_weight is not None:
            w_ref, w_gen_1, w_gen_2 = tf.split(
                sample_weight[: batch_size * 3], 3, axis=0
            )
        else:
            w_ref, w_gen_1, w_gen_2 = tf.split(
                tf.ones(shape=(batch_size * 3,)), 3, axis=0
            )

        return (
            (x_ref, y_ref, w_ref),
            (x_gen_1, y_gen_1, w_gen_1),
            (x_gen_2, y_gen_2, w_gen_2),
        )

    @staticmethod
    def _standard_loss_func(
        critic,
        trainset_ref,
        trainset_gen_1,
        trainset_gen_2,
        training_critic=False,
        generator_loss=True,
    ) -> tf.Tensor:
        x_ref, y_ref, w_ref = trainset_ref
        x_gen_1, y_gen_1, w_gen_1 = trainset_gen_1
        x_gen_2, y_gen_2, w_gen_2 = trainset_gen_2

        x_concat_1 = tf.concat([x_ref, x_gen_1], axis=0)
        y_concat_1 = tf.concat([y_ref, y_gen_1], axis=0)
        x_concat_2 = tf.concat([x_gen_2, x_gen_2], axis=0)
        y_concat_2 = tf.concat([y_gen_2, y_gen_2], axis=0)

        c_out = critic(
            (x_concat_1, y_concat_1), (x_concat_2, y_concat_2), training=training_critic
        )
        c_ref, c_gen = tf.split(c_out, 2, axis=0)

        real_loss = tf.reduce_sum(w_ref * w_gen_2 * c_ref) / tf.reduce_sum(
            w_ref * w_gen_2
        )
        fake_loss = tf.reduce_sum(w_gen_1 * w_gen_2 * c_gen) / tf.reduce_sum(
            w_gen_1 * w_gen_2
        )

        if generator_loss:
            return real_loss - fake_loss
        else:
            return fake_loss - real_loss

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        return self._standard_loss_func(
            critic=self._critic,
            trainset_ref=trainset_ref,
            trainset_gen_1=trainset_gen_1,
            trainset_gen_2=trainset_gen_2,
            training_critic=False,
            generator_loss=True,
        )

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        d_loss = self._standard_loss_func(
            critic=self._critic,
            trainset_ref=trainset_ref,
            trainset_gen_1=trainset_gen_1,
            trainset_gen_2=trainset_gen_2,
            training_critic=training,
            generator_loss=False,
        )
        lip_reg = self._lipschitz_regularization(
            self._critic, x, y, sample_weight, training_critic=training
        )
        return d_loss + lip_reg

    def _compute_r_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen, _ = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)

        r_out = self._referee((x_concat, y_concat), training=training)
        r_ref, r_gen = tf.split(r_out, 2, axis=0)

        real_loss = self._referee_loss(tf.ones_like(r_ref), r_ref, sample_weight=w_ref)
        fake_loss = self._referee_loss(tf.zeros_like(r_gen), r_gen, sample_weight=w_gen)
        return (real_loss + fake_loss) / 2.0

    def _lipschitz_regularization(
        self, critic, x, y, sample_weight=None, training_critic=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen_1, trainset_gen_2 = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return compute_CriticGradientPenalty(
            critic=critic,
            trainset_ref=trainset_ref,
            trainset_gen_1=trainset_gen_1,
            trainset_gen_2=trainset_gen_2,
            training_critic=training_critic,
            lipschitz_penalty=self._lipschitz_penalty,
            lipschitz_penalty_strategy=self._lipschitz_penalty_strategy,
            lipschitz_constant=LIPSCHITZ_CONSTANT,
        )
