import os

import pytest
import tensorflow as tf
from tensorflow import keras

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/res-generator"

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))


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
    output = model(x)
    model.summary()
    test_shape = [x.shape[0]]
    test_shape.append(model.output_dim)
    assert output.shape == tuple(test_shape)
    assert isinstance(model.export_model, keras.Model)


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
    adam = keras.optimizers.Adam(learning_rate=0.001)
    mse = keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse, metrics=["mae"])
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, sample_weight):
    adam = keras.optimizers.Adam(learning_rate=0.001)
    mse = keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse, metrics=["mae"])
    model.evaluate(x, sample_weight=sample_weight)


def test_model_generate(model):
    no_seed_out = model.generate(x, seed=None)
    comparison = no_seed_out.numpy() != model.generate(x, seed=None).numpy()
    assert comparison.all()
    seed_out = model.generate(x, seed=42)
    comparison = seed_out.numpy() == model.generate(x, seed=42).numpy()
    assert comparison.all()
    comparison = seed_out.numpy() != model.generate(x, seed=24).numpy()
    assert comparison.any()


def test_model_export(model):
    out, latent_sample = model.generate(x, return_latent_sample=True)
    keras.models.save_model(model.export_model, export_dir, save_format="tf")
    model_reloaded = keras.models.load_model(export_dir)
    x_reloaded = tf.concat([x, latent_sample], axis=-1)
    out_reloaded = model_reloaded(x_reloaded)
    comparison = out.numpy() == out_reloaded.numpy()
    assert comparison.all()
