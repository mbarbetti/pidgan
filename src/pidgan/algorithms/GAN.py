import tensorflow as tf
from tensorflow import keras

from pidgan.players.discriminators import Discriminator
from pidgan.players.generators import Generator
from pidgan.utils.checks import checkMetrics, checkOptimizer

MIN_LOG_VALUE = 1e-8
MAX_LOG_VALUE = 1.0


class GAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        referee=None,
        use_original_loss=True,
        injected_noise_stddev=0.0,
        feature_matching_penalty=0.0,
        referee_from_logits=None,
        referee_label_smoothing=None,
        name="GAN",
        dtype=None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        self._loss_name = "GAN original loss"

        # Generator
        if not isinstance(generator, Generator):
            raise TypeError(
                f"`generator` should be a pidgan's `Generator`, "
                f"instead {type(generator)} passed"
            )
        self._generator = generator

        # Discriminator
        if not isinstance(discriminator, Discriminator):
            raise TypeError(
                f"`discriminator` should be a pidgan's `Discriminator`, "
                f"instead {type(discriminator)} passed"
            )
        self._discriminator = discriminator

        # Referee network
        if referee is not None:
            if not isinstance(referee, Discriminator):
                raise TypeError(
                    f"`referee` should be a pidgan's `Discriminator`, "
                    f"instead {type(referee)} passed"
                )
            self._referee = referee
        else:
            self._referee = None

        # Flag to use the original loss
        assert isinstance(use_original_loss, bool)
        self._use_original_loss = use_original_loss

        # Noise standard deviation
        assert isinstance(injected_noise_stddev, (int, float))
        assert injected_noise_stddev >= 0.0
        self._inj_noise_std = float(injected_noise_stddev)

        # Feature matching penalty
        assert isinstance(feature_matching_penalty, (int, float))
        assert feature_matching_penalty >= 0.0
        self._feature_matching_penalty = float(feature_matching_penalty)

        # TensorFlow BinaryCrossentropy
        if (referee_from_logits is not None) and (referee_label_smoothing is not None):
            self._bce_loss = keras.losses.BinaryCrossentropy(
                from_logits=referee_from_logits, label_smoothing=referee_label_smoothing
            )
            self._referee_from_logits = bool(referee_from_logits)
            self._referee_label_smoothing = float(referee_label_smoothing)
            self._referee_loss_name = "Binary cross-entropy"
        else:
            self._bce_loss = None
            self._referee_from_logits = None
            self._referee_label_smoothing = None
            self._referee_loss_name = self._loss_name

    def call(self, x, y=None) -> tuple:
        g_out = self._generator(x)
        d_out_gen = self._discriminator((x, g_out))
        if y is None:
            if self._referee is not None:
                r_out_gen = self._referee((x, g_out))
                return g_out, d_out_gen, r_out_gen
            else:
                return g_out, d_out_gen
        else:
            d_out_ref = self._discriminator((x, y))
            if self._referee is not None:
                r_out_gen = self._referee((x, g_out))
                r_out_ref = self._referee((x, y))
                return g_out, (d_out_gen, d_out_ref), (r_out_gen, r_out_ref)
            else:
                return g_out, (d_out_gen, d_out_ref)

    def summary(self, **kwargs) -> None:
        print("_" * 65)
        self._generator.summary(**kwargs)
        self._discriminator.summary(**kwargs)
        if self._referee is not None:
            self._referee.summary(**kwargs)

    def compile(
        self,
        metrics=None,
        generator_optimizer="rmsprop",
        discriminator_optimizer="rmsprop",
        referee_optimizer="rmsprop",
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        referee_upds_per_batch=1,
    ) -> None:
        super().compile(weighted_metrics=[])

        # Loss metrics
        self._g_loss = keras.metrics.Mean(name="g_loss")
        self._d_loss = keras.metrics.Mean(name="d_loss")
        if self._referee is not None:
            self._r_loss = keras.metrics.Mean(name="r_loss")
        self._metrics = checkMetrics(metrics)

        # Optimizers
        self._g_opt = checkOptimizer(generator_optimizer)
        self._d_opt = checkOptimizer(discriminator_optimizer)
        if self._referee is not None:
            self._r_opt = checkOptimizer(referee_optimizer)
        else:
            self._r_opt = None

        # Generator updates per batch
        assert isinstance(generator_upds_per_batch, (int, float))
        assert generator_upds_per_batch >= 1
        self._g_upds_per_batch = int(generator_upds_per_batch)

        # Discriminator updates per batch
        assert isinstance(discriminator_upds_per_batch, (int, float))
        assert discriminator_upds_per_batch >= 1
        self._d_upds_per_batch = int(discriminator_upds_per_batch)

        # Referee updates per batch
        if self._referee is not None:
            assert isinstance(referee_upds_per_batch, (int, float))
            assert referee_upds_per_batch >= 1
            self._r_upds_per_batch = int(referee_upds_per_batch)
        else:
            self._r_upds_per_batch = None

    def train_step(self, data) -> dict:
        x, y, sample_weight = self._unpack_data(data)

        for _ in range(self._d_upds_per_batch):
            self._d_train_step(x, y, sample_weight)
        for _ in range(self._g_upds_per_batch):
            self._g_train_step(x, y, sample_weight)
        if self._referee is not None:
            for _ in range(self._r_upds_per_batch):
                self._r_train_step(x, y, sample_weight)

        train_dict = dict(g_loss=self._g_loss.result(), d_loss=self._d_loss.result())
        if self._referee is not None:
            train_dict.update(dict(r_loss=self._r_loss.result()))
        if self._metrics is not None:
            g_out = self._generator(x, training=False)
            x_concat = tf.concat([x, x], axis=0)
            y_concat = tf.concat([y, g_out], axis=0)
            d_out = self._discriminator((x_concat, y_concat), training=False)
            d_ref, d_gen = tf.split(d_out, 2, axis=0)
            for metric in self._metrics:
                metric.update_state(
                    y_true=d_ref, y_pred=d_gen, sample_weight=sample_weight
                )
                train_dict.update({metric.name: metric.result()})
        return train_dict

    @staticmethod
    def _unpack_data(data) -> tuple:
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        return x, y, sample_weight

    def _g_train_step(self, x, y, sample_weight=None) -> None:
        with tf.GradientTape() as tape:
            loss = self._compute_g_loss(x, y, sample_weight, training=True)
            loss += self._compute_feature_matching(x, y, sample_weight, training=True)

        trainable_vars = self._generator.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self._g_opt.apply_gradients(zip(gradients, trainable_vars))

        threshold = self._compute_threshold(self._discriminator, x, y, sample_weight)
        self._g_loss.update_state(loss + threshold)

    def _d_train_step(self, x, y, sample_weight=None) -> None:
        with tf.GradientTape() as tape:
            loss = self._compute_d_loss(x, y, sample_weight, training=True)

        trainable_vars = self._discriminator.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self._d_opt.apply_gradients(zip(gradients, trainable_vars))

        threshold = self._compute_threshold(self._discriminator, x, y, sample_weight)
        self._d_loss.update_state(loss - threshold)

    def _r_train_step(self, x, y, sample_weight=None) -> None:
        with tf.GradientTape() as tape:
            loss = self._compute_r_loss(x, y, sample_weight, training=True)

        trainable_vars = self._referee.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self._r_opt.apply_gradients(zip(gradients, trainable_vars))

        threshold = self._compute_threshold(self._referee, x, y, sample_weight)
        self._r_loss.update_state(loss - threshold)

    def _prepare_trainset(
        self, x, y, sample_weight=None, training_generator=True
    ) -> tuple:
        batch_size = tf.cast(tf.shape(x)[0] / 2, tf.int32)
        x_ref, x_gen = tf.split(x[: batch_size * 2], 2, axis=0)
        y_ref = y[:batch_size]
        y_gen = self._generator(x_gen, training=training_generator)

        if sample_weight is not None:
            w_ref, w_gen = tf.split(sample_weight[: batch_size * 2], 2, axis=0)
        else:
            w_ref, w_gen = tf.split(tf.ones(shape=(batch_size * 2,)), 2, axis=0)

        return (x_ref, y_ref, w_ref), (x_gen, y_gen, w_gen)

    @staticmethod
    def _standard_loss_func(
        model,
        trainset_ref,
        trainset_gen,
        inj_noise_std=0.0,
        model_training=False,
        original_loss=True,
    ) -> tf.Tensor:
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        if inj_noise_std > 0.0:
            rnd_noise = tf.random.normal(
                shape=(tf.shape(y_ref)[0] * 2, tf.shape(y_ref)[1]),
                mean=0.0,
                stddev=inj_noise_std,
                dtype=y_ref.dtype,
            )
        else:
            rnd_noise = 0.0

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)

        m_out = model((x_concat, y_concat + rnd_noise), training=model_training)
        m_ref, m_gen = tf.split(m_out, 2, axis=0)

        real_loss = tf.reduce_sum(
            w_ref[:, None]
            * tf.math.log(tf.clip_by_value(m_ref, MIN_LOG_VALUE, MAX_LOG_VALUE))
        ) / tf.reduce_sum(w_ref)
        if original_loss:
            fake_loss = tf.reduce_sum(
                w_gen[:, None]
                * tf.math.log(
                    tf.clip_by_value(1.0 - m_gen, MIN_LOG_VALUE, MAX_LOG_VALUE)
                )
            ) / tf.reduce_sum(w_gen)
        else:
            fake_loss = tf.reduce_sum(
                -w_gen[:, None]
                * tf.math.log(tf.clip_by_value(m_gen, MIN_LOG_VALUE, MAX_LOG_VALUE))
            )
        return real_loss + fake_loss

    def _compute_g_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=training
        )
        return self._standard_loss_func(
            model=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            inj_noise_std=self._inj_noise_std,
            model_training=False,
            original_loss=self._use_original_loss,
        )

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return -self._standard_loss_func(
            model=self._discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            inj_noise_std=self._inj_noise_std,
            model_training=training,
            original_loss=True,
        )

    def _bce_loss_func(
        self, model, trainset_ref, trainset_gen, inj_noise_std=0.0, model_training=True
    ) -> tf.Tensor:
        x_ref, y_ref, w_ref = trainset_ref
        x_gen, y_gen, w_gen = trainset_gen

        if inj_noise_std > 0.0:
            rnd_noise = tf.random.normal(
                shape=(tf.shape(y_ref)[0] * 2, tf.shape(y_ref)[1]),
                mean=0.0,
                stddev=inj_noise_std,
                dtype=y_ref.dtype,
            )
        else:
            rnd_noise = 0.0

        x_concat = tf.concat([x_ref, x_gen], axis=0)
        y_concat = tf.concat([y_ref, y_gen], axis=0)

        m_out = model((x_concat, y_concat + rnd_noise), training=model_training)
        m_ref, m_gen = tf.split(m_out, 2, axis=0)

        real_loss = self._bce_loss(tf.ones_like(m_ref), m_ref, sample_weight=w_ref)
        fake_loss = self._bce_loss(tf.zeros_like(m_gen), m_gen, sample_weight=w_gen)
        return (real_loss + fake_loss) / 2.0

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
                inj_noise_std=0.0,
                model_training=training,
                original_loss=True,
            )

    def _compute_feature_matching(
        self, x, y, sample_weight=None, training=True
    ) -> tf.Tensor:
        if self._feature_matching_penalty > 0.0:
            trainset_ref, trainset_gen = self._prepare_trainset(
                x, y, sample_weight, training_generator=training
            )
            x_ref, y_ref, _ = trainset_ref
            x_gen, y_gen, _ = trainset_gen

            rnd_noise = 0.0
            if self._inj_noise_std is not None:
                if self._inj_noise_std > 0.0:
                    rnd_noise = tf.random.normal(
                        shape=(tf.shape(y_ref)[0] * 2, tf.shape(y_ref)[1]),
                        mean=0.0,
                        stddev=self._inj_noise_std,
                        dtype=y_ref.dtype,
                    )

            x_concat = tf.concat([x_ref, x_gen], axis=0)
            y_concat = tf.concat([y_ref, y_gen], axis=0)

            d_feat_out = self._discriminator.hidden_feature(
                (x_concat, y_concat + rnd_noise)
            )
            d_feat_ref, d_feat_gen = tf.split(d_feat_out, 2, axis=0)

            feat_match_term = tf.norm(d_feat_ref - d_feat_gen, axis=-1) ** 2
            return self._feature_matching_penalty * tf.reduce_mean(feat_match_term)
        else:
            return 0.0

    def _prepare_trainset_threshold(self, x, y, sample_weight=None) -> tuple:
        batch_size = tf.cast(tf.shape(x)[0] / 2, tf.int32)
        x_ref_1, x_ref_2 = tf.split(x[: batch_size * 2], 2, axis=0)
        y_ref_1, y_ref_2 = tf.split(y[: batch_size * 2], 2, axis=0)

        if sample_weight is not None:
            w_ref_1, w_ref_2 = tf.split(sample_weight[: batch_size * 2], 2, axis=0)
        else:
            w_ref_1, w_ref_2 = tf.split(tf.ones(shape=(batch_size * 2,)), 2, axis=0)

        return (x_ref_1, y_ref_1, w_ref_1), (x_ref_2, y_ref_2, w_ref_2)

    def _compute_threshold(self, model, x, y, sample_weight=None) -> tf.Tensor:
        trainset_ref_1, trainset_ref_2 = self._prepare_trainset_threshold(
            x, y, sample_weight
        )
        return -self._standard_loss_func(
            model=model,
            trainset_ref=trainset_ref_1,
            trainset_gen=trainset_ref_2,
            inj_noise_std=0.0,
            model_training=False,
            original_loss=True,
        )

    def test_step(self, data) -> dict:
        x, y, sample_weight = self._unpack_data(data)

        threshold = self._compute_threshold(self._discriminator, x, y, sample_weight)

        g_loss = self._compute_g_loss(x, y, sample_weight, training=False)
        self._g_loss.update_state(g_loss + threshold)

        d_loss = self._compute_d_loss(x, y, sample_weight, training=False)
        self._d_loss.update_state(d_loss - threshold)

        if self._referee is not None:
            r_loss = self._compute_r_loss(x, y, sample_weight, training=False)
            threshold = self._compute_threshold(self._referee, x, y, sample_weight)
            self._r_loss.update_state(r_loss - threshold)

        train_dict = dict(g_loss=self._g_loss.result(), d_loss=self._d_loss.result())
        if self._referee is not None:
            train_dict.update(dict(r_loss=self._r_loss.result()))
        if self._metrics is not None:
            g_out = self._generator(x, training=False)
            x_concat = tf.concat([x, x], axis=0)
            y_concat = tf.concat([y, g_out], axis=0)
            d_out = self._discriminator((x_concat, y_concat), training=False)
            d_ref, d_gen = tf.split(d_out, 2, axis=0)
            for metric in self._metrics:
                metric.update_state(
                    y_true=d_ref, y_pred=d_gen, sample_weight=sample_weight
                )
                train_dict.update({metric.name: metric.result()})
        return train_dict

    def generate(self, x, seed=None) -> tf.Tensor:
        return self._generator.generate(x, seed=seed)

    @property
    def loss_name(self) -> str:
        return self._loss_name

    @property
    def generator(self) -> Generator:
        return self._generator

    @property
    def discriminator(self) -> Discriminator:
        return self._discriminator

    @property
    def referee(self):  # TODO: add Union[None, Discriminator]
        return self._referee

    @property
    def referee_loss_name(self):  # TODO: add Union[None, str]
        return self._referee_loss_name

    @property
    def use_original_loss(self) -> bool:
        return self._use_original_loss

    @property
    def injected_noise_stddev(self) -> float:
        return self._inj_noise_std

    @property
    def feature_matching_penalty(self) -> float:
        return self._feature_matching_penalty

    @property
    def referee_from_logits(self):  # TODO: add Union[None, bool]
        return self._referee_from_logits

    @property
    def referee_label_smoothing(self):  # TODO: add Union[None, float]
        return self._referee_label_smoothing

    @property
    def metrics(self) -> list:
        reset_states = [self._g_loss, self._d_loss]
        if self._referee is not None:
            reset_states += [self._r_loss]
        if self._metrics is not None:
            reset_states += self._metrics
        return reset_states

    @property
    def generator_optimizer(self) -> keras.optimizers.Optimizer:
        return self._g_opt

    @property
    def discriminator_optimizer(self) -> keras.optimizers.Optimizer:
        return self._d_opt

    @property
    def referee_optimizer(self):  # TODO: add Union[None, Optimizer]
        return self._r_opt

    @property
    def generator_upds_per_batch(self) -> int:
        return self._g_upds_per_batch

    @property
    def discriminator_upds_per_batch(self) -> int:
        return self._d_upds_per_batch

    @property
    def referee_upds_per_batch(self):  # TODO: add Union[None, int]
        return self._r_upds_per_batch
