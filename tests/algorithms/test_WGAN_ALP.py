import pytest
import tensorflow as tf
from tensorflow import keras

from pidgan.players.discriminators import Discriminator
from pidgan.players.generators import Generator

CHUNK_SIZE = int(1e4)

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))

gen = Generator(
    output_dim=y.shape[1],
    latent_dim=64,
    num_hidden_layers=4,
    mlp_hidden_units=32,
    dropout_rate=0.1,
    output_activation=None,
)

disc = Discriminator(
    output_dim=1,
    num_hidden_layers=4,
    mlp_hidden_units=32,
    dropout_rate=0.0,
    output_activation=None,
)

ref = Discriminator(
    output_dim=1,
    num_hidden_layers=2,
    mlp_hidden_units=32,
    dropout_rate=0.0,
    output_activation=None,
)


@pytest.fixture
def model():
    from pidgan.algorithms import WGAN_ALP

    gan = WGAN_ALP(
        generator=gen,
        discriminator=disc,
        referee=ref,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="one-sided",
        feature_matching_penalty=0.0,
        referee_from_logits=True,
        referee_label_smoothing=0.0,
    )
    return gan


###########################################################################


def test_model_configuration(model):
    from pidgan.algorithms import WGAN_ALP
    from pidgan.players.discriminators import Discriminator
    from pidgan.players.generators import Generator

    assert isinstance(model, WGAN_ALP)
    assert isinstance(model.loss_name, str)
    assert isinstance(model.generator, Generator)
    assert isinstance(model.discriminator, Discriminator)
    assert isinstance(model.referee, Discriminator)
    assert isinstance(model.referee_loss_name, str)
    assert isinstance(model.lipschitz_penalty, float)
    assert isinstance(model.lipschitz_penalty_strategy, str)
    assert isinstance(model.feature_matching_penalty, float)
    if model.referee_from_logits is not None:
        assert isinstance(model.referee_from_logits, bool)
    if model.referee_label_smoothing is not None:
        assert isinstance(model.referee_label_smoothing, float)


@pytest.mark.parametrize("referee", [ref, None])
def test_model_use(referee):
    from pidgan.algorithms import WGAN_ALP

    model = WGAN_ALP(
        generator=gen,
        discriminator=disc,
        referee=referee,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="one-sided",
        feature_matching_penalty=0.0,
        referee_from_logits=None,
        referee_label_smoothing=None,
    )
    outputs = model(x, y)
    if referee is not None:
        g_output, d_outputs, r_outputs = outputs
    else:
        g_output, d_outputs = outputs
    model.summary()

    test_g_shape = [y.shape[0]]
    test_g_shape.append(model.generator.output_dim)
    assert g_output.shape == tuple(test_g_shape)

    test_d_shape = [y.shape[0]]
    test_d_shape.append(model.discriminator.output_dim)
    d_output_gen, d_output_ref = d_outputs
    assert d_output_gen.shape == tuple(test_d_shape)
    assert d_output_ref.shape == tuple(test_d_shape)

    if referee is not None:
        test_r_shape = [y.shape[0]]
        test_r_shape.append(model.referee.output_dim)
        r_output_gen, r_output_ref = r_outputs
        assert r_output_gen.shape == tuple(test_r_shape)
        assert r_output_ref.shape == tuple(test_r_shape)


@pytest.mark.parametrize("metrics", [["bce"], None])
def test_model_compilation(model, metrics):
    g_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    d_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    r_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        metrics=metrics,
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        referee_optimizer=r_opt,
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        referee_upds_per_batch=1,
        virtual_direction_upds=1,
    )
    assert isinstance(model.metrics, list)
    assert isinstance(model.generator_optimizer, keras.optimizers.Optimizer)
    assert isinstance(model.discriminator_optimizer, keras.optimizers.Optimizer)
    assert isinstance(model.referee_optimizer, keras.optimizers.Optimizer)
    assert isinstance(model.generator_upds_per_batch, int)
    assert isinstance(model.discriminator_upds_per_batch, int)
    assert isinstance(model.referee_upds_per_batch, int)
    assert isinstance(model.virtual_direction_upds, int)


@pytest.mark.parametrize("referee", [ref, None])
@pytest.mark.parametrize("sample_weight", [w, None])
@pytest.mark.parametrize(
    "loss_options", [("one-sided", True, 0.0), ("two-sided", None, None)]
)
def test_model_train(referee, sample_weight, loss_options):
    from pidgan.algorithms import WGAN_ALP

    if sample_weight is not None:
        slices = (x, y, w)
    else:
        slices = (x, y)
    dataset = (
        tf.data.Dataset.from_tensor_slices(slices)
        .batch(batch_size=512, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    penalty_strategy, from_logits, label_smoothing = loss_options

    model = WGAN_ALP(
        generator=gen,
        discriminator=disc,
        referee=referee,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy=penalty_strategy,
        feature_matching_penalty=1.0,
        referee_from_logits=from_logits,
        referee_label_smoothing=label_smoothing,
    )
    g_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    d_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    r_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        metrics=None,
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        referee_optimizer=r_opt,
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        referee_upds_per_batch=1,
    )
    model.fit(dataset, epochs=1)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, sample_weight):
    g_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    d_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    r_opt = keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        metrics=None,
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        referee_optimizer=r_opt,
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        referee_upds_per_batch=1,
    )
    model.evaluate(x, y, sample_weight=sample_weight)
