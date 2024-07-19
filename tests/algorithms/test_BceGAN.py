import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import pytest
import warnings
import numpy as np
import tensorflow as tf

from pidgan.players.classifiers import AuxClassifier
from pidgan.players.discriminators import AuxDiscriminator
from pidgan.players.generators import ResGenerator
from pidgan.metrics import BinaryCrossentropy as BCE

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

x = np.random.normal(size=(CHUNK_SIZE, 4)).astype("float32")
y = np.random.normal(size=(CHUNK_SIZE, 8)).astype("float32")
w = np.random.uniform(size=(CHUNK_SIZE,)).astype("float32")

gen = ResGenerator(
    output_dim=y.shape[1],
    latent_dim=64,
    num_hidden_layers=4,
    mlp_hidden_units=32,
    mlp_dropout_rates=0.1,
    output_activation=None,
)

disc = AuxDiscriminator(
    output_dim=1,
    aux_features=["0 + 1", "2 - 3"],
    num_hidden_layers=4,
    mlp_hidden_units=32,
    mlp_dropout_rates=0.0,
    output_activation="sigmoid",
)

ref = AuxClassifier(
    aux_features=["0 + 1", "2 - 3"],
    num_hidden_layers=2,
    mlp_hidden_units=32,
    mlp_dropout_rates=0.0,
)


@pytest.fixture
def model():
    from pidgan.algorithms import BceGAN

    gan = BceGAN(
        generator=gen,
        discriminator=disc,
        from_logits=False,
        label_smoothing=0.0,
        injected_noise_stddev=0.1,
        feature_matching_penalty=0.0,
        referee=ref,
    )
    return gan


###########################################################################


def test_model_configuration(model):
    from pidgan.algorithms import BceGAN
    from pidgan.players.discriminators import Discriminator
    from pidgan.players.generators import Generator

    assert isinstance(model, BceGAN)
    assert isinstance(model.loss_name, str)
    assert isinstance(model.generator, Generator)
    assert isinstance(model.discriminator, Discriminator)
    assert isinstance(model.from_logits, bool)
    assert isinstance(model.label_smoothing, float)
    assert isinstance(model.injected_noise_stddev, float)
    assert isinstance(model.feature_matching_penalty, float)
    assert isinstance(model.referee, Discriminator)


@pytest.mark.parametrize("referee", [ref, None])
def test_model_use(referee):
    from pidgan.algorithms import BceGAN

    model = BceGAN(
        generator=gen,
        discriminator=disc,
        from_logits=False,
        label_smoothing=0.0,
        injected_noise_stddev=0.1,
        feature_matching_penalty=0.0,
        referee=referee,
    )
    out = model(x, y)
    if referee is not None:
        g_out, d_out, r_out = out
    else:
        g_out, d_out = out
    model.summary()

    test_g_shape = [y.shape[0]]
    test_g_shape.append(model.generator.output_dim)
    assert g_out.shape == tuple(test_g_shape)

    test_d_shape = [y.shape[0]]
    test_d_shape.append(model.discriminator.output_dim)
    d_out_gen, d_out_ref = d_out
    assert d_out_gen.shape == tuple(test_d_shape)
    assert d_out_ref.shape == tuple(test_d_shape)

    if referee is not None:
        test_r_shape = [y.shape[0]]
        test_r_shape.append(model.referee.output_dim)
        r_out_gen, r_out_ref = r_out
        assert r_out_gen.shape == tuple(test_r_shape)
        assert r_out_ref.shape == tuple(test_r_shape)


@pytest.mark.parametrize("build_first", [True, False])
@pytest.mark.parametrize("metrics", [["bce"], [BCE()], None])
def test_model_compilation(model, build_first, metrics):
    import keras as k

    if build_first:
        model(x, y)  # to build the model

    g_opt = k.optimizers.RMSprop(learning_rate=0.001)
    d_opt = k.optimizers.RMSprop(learning_rate=0.001)
    r_opt = k.optimizers.RMSprop(learning_rate=0.001)

    with warnings.catch_warnings(record=True) as w:
        if build_first:
            warnings.simplefilter("always")
        else:
            warnings.simplefilter("ignore")

        model.compile(
            metrics=metrics,
            generator_optimizer=g_opt,
            discriminator_optimizer=d_opt,
            generator_upds_per_batch=1,
            discriminator_upds_per_batch=1,
            referee_optimizer=r_opt,
            referee_upds_per_batch=1,
        )
        if build_first and metrics is not None:
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "`compile()`" in str(w[-1].message)

    assert isinstance(model.metrics, list)
    assert isinstance(model.generator_optimizer, k.optimizers.Optimizer)
    assert isinstance(model.discriminator_optimizer, k.optimizers.Optimizer)
    assert isinstance(model.generator_upds_per_batch, int)
    assert isinstance(model.discriminator_upds_per_batch, int)
    assert isinstance(model.referee_optimizer, k.optimizers.Optimizer)
    assert isinstance(model.referee_upds_per_batch, int)

    if not build_first:
        model(x, y)  # to build the model
        if metrics is None:
            assert len(model.metrics) == 3  # g_loss, d_loss, r_loss
        else:
            assert len(model.metrics) == 4  # losses + bce
    else:
        assert len(model.metrics) == 3  # g_loss, d_loss, r_loss


@pytest.mark.parametrize("referee", [ref, None])
@pytest.mark.parametrize("sample_weight", [w, None])
@pytest.mark.parametrize("build_first", [True, False])
def test_model_train(referee, sample_weight, build_first):
    import keras as k
    from pidgan.algorithms import BceGAN

    if sample_weight is not None:
        slices = (x, y, w)
    else:
        slices = (x, y)
    train_ds = (
        tf.data.Dataset.from_tensor_slices(slices)
        .shuffle(buffer_size=int(0.1 * CHUNK_SIZE))
        .batch(batch_size=BATCH_SIZE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(slices)
        .shuffle(buffer_size=int(0.1 * CHUNK_SIZE))
        .batch(batch_size=BATCH_SIZE)
    )

    model = BceGAN(
        generator=gen,
        discriminator=disc,
        from_logits=False,
        label_smoothing=0.0,
        injected_noise_stddev=0.1,
        feature_matching_penalty=1.0,
        referee=referee,
    )
    if build_first:
        model(x, y)  # to build the model

    g_opt = k.optimizers.RMSprop(learning_rate=0.001)
    d_opt = k.optimizers.RMSprop(learning_rate=0.001)
    r_opt = k.optimizers.RMSprop(learning_rate=0.001)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.compile(
            metrics=["bce"],
            generator_optimizer=g_opt,
            discriminator_optimizer=d_opt,
            generator_upds_per_batch=1,
            discriminator_upds_per_batch=1,
            referee_optimizer=r_opt,
            referee_upds_per_batch=1,
        )
    if not build_first:
        model(x, y)  # to build the model

    train = model.fit(
        train_ds.take(BATCH_SIZE),
        epochs=2,
        validation_data=val_ds.take(BATCH_SIZE),
    )
    states = train.history.keys()

    if not build_first:
        if referee is not None:
            assert len(states) == 8  # 2x (g_loss + d_loss + r_loss + bce)
        else:
            assert len(states) == 6  # 2x (g_loss + d_loss + bce)
    else:
        if referee is not None:
            assert len(states) == 6  # 2x (g_loss + d_loss + r_loss)
        else:
            assert len(states) == 4  # 2x (g_loss + d_loss)

    for s in states:
        for entry in train.history[s]:
            assert isinstance(entry, (int, float))

    if not build_first:
        for key in ["g_loss", "d_loss", "bce"]:
            comparison = (
                np.array(train.history[key]) != np.array(train.history[f"val_{key}"])
            )
            assert comparison.all()


@pytest.mark.parametrize("metrics", [["bce"], [BCE()], None])
@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, metrics, sample_weight):
    import keras as k
    
    g_opt = k.optimizers.RMSprop(learning_rate=0.001)
    d_opt = k.optimizers.RMSprop(learning_rate=0.001)
    r_opt = k.optimizers.RMSprop(learning_rate=0.001)
    model.compile(
        metrics=metrics,
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        referee_optimizer=r_opt,
        referee_upds_per_batch=1,
    )
    model.evaluate(x, y, sample_weight=sample_weight)
