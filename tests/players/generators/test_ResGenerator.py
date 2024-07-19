import os

import pytest
import keras as k
import numpy as np
import tensorflow as tf

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/res-generator"

x = np.random.normal(size=(CHUNK_SIZE, 4)).astype("float32")
y = np.random.normal(size=(CHUNK_SIZE, 8)).astype("float32")
w = np.random.uniform(size=(CHUNK_SIZE,)).astype("float32")


@pytest.fixture
def model():
    from pidgan.players.generators import ResGenerator

    gen = ResGenerator(
        output_dim=y.shape[1],
        latent_dim=64,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        output_activation=None,
    )
    return gen


###########################################################################


def test_model_configuration(model):
    from pidgan.players.generators import ResGenerator

    assert isinstance(model, ResGenerator)
    assert isinstance(model.output_dim, int)
    assert isinstance(model.latent_dim, int)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, int)
    assert isinstance(model.mlp_dropout_rates, float)
    # assert isinstance(model.output_activation, str)


@pytest.mark.parametrize("output_activation", ["linear", None])
def test_model_use(output_activation):
    from pidgan.players.generators import ResGenerator

    model = ResGenerator(
        output_dim=y.shape[1],
        latent_dim=64,
        num_hidden_layers=3,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        output_activation=output_activation,
    )

    out = model(x)
    model.summary()
    test_shape = [x.shape[0]]
    test_shape.append(model.output_dim)
    assert out.shape == tuple(test_shape)
    assert isinstance(model.plain_keras, k.Model)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_train(model, sample_weight):
    if sample_weight is not None:
        slices = (x, y, w)
    else:
        slices = (x, y)
    dataset = (
        tf.data.Dataset.from_tensor_slices(slices)
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=0.001),
        loss=k.losses.MeanSquaredError(),
        metrics=["mae"],
    )
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, sample_weight):
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=0.001),
        loss=k.losses.MeanSquaredError(),
        metrics=["mae"],
    )
    model.evaluate(x, y, sample_weight=sample_weight)


def test_model_generate(model):
    no_seed_out = model.generate(x[:BATCH_SIZE], seed=None)
    comparison = (
        no_seed_out.numpy() != model.generate(x[:BATCH_SIZE], seed=None).numpy()
    )
    assert comparison.all()
    seed_out = model.generate(x[:BATCH_SIZE], seed=42)
    comparison = seed_out.numpy() == model.generate(x[:BATCH_SIZE], seed=42).numpy()
    assert comparison.all()
    comparison = seed_out.numpy() != model.generate(x[:BATCH_SIZE], seed=24).numpy()
    assert comparison.any()


def test_model_export(model):
    out, latent_sample = model.generate(x[:BATCH_SIZE], return_latent_sample=True)

    v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]
    if v_major == 3 and v_minor >= 0:
        model.plain_keras.export(export_dir)
        model_reloaded = k.layers.TFSMLayer(export_dir, call_endpoint="serve")
    else:
        k.models.save_model(model.plain_keras, export_dir, save_format="tf")
        model_reloaded = k.models.load_model(export_dir)

    x_reloaded = tf.concat([x[:BATCH_SIZE], latent_sample], axis=-1)
    out_reloaded = model_reloaded(x_reloaded)
    comparison = out.numpy() == out_reloaded.numpy()
    assert comparison.all()
