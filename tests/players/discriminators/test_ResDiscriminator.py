import os

import pytest
import tensorflow as tf
from tensorflow import keras

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/res-discriminator"

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))
labels = tf.random.uniform(shape=(CHUNK_SIZE,), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, x.dtype)


@pytest.fixture
def model():
    from pidgan.players.discriminators import ResDiscriminator

    disc = ResDiscriminator(
        output_dim=1,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        output_activation="sigmoid",
    )
    return disc


###########################################################################


def test_model_configuration(model):
    from pidgan.players.discriminators import ResDiscriminator

    assert isinstance(model, ResDiscriminator)
    assert isinstance(model.output_dim, int)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, int)
    assert isinstance(model.mlp_dropout_rates, float)
    # assert isinstance(model.output_activation, str)


@pytest.mark.parametrize("output_activation", ["sigmoid", None])
def test_model_use(output_activation):
    from pidgan.players.discriminators import ResDiscriminator

    model = ResDiscriminator(
        output_dim=1,
        num_hidden_layers=3,
        mlp_hidden_units=128,
        mlp_dropout_rates=0.0,
        output_activation=output_activation,
    )
    out = model((x, y))
    model.summary()
    test_shape = [x.shape[0]]
    test_shape.append(model.output_dim)
    assert out.shape == tuple(test_shape)
    hidden_feat = model.hidden_feature((x, y))
    test_shape = [x.shape[0]]
    test_shape.append(model.mlp_hidden_units)
    assert hidden_feat.shape == tuple(test_shape)
    assert isinstance(model.export_model, keras.Model)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_train(model, sample_weight):
    if sample_weight is not None:
        slices = ((x, y), labels, w)
    else:
        slices = ((x, y), labels)
    dataset = (
        tf.data.Dataset.from_tensor_slices(slices)
        .batch(batch_size=BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    adam = keras.optimizers.Adam(learning_rate=0.001)
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam, loss=bce, metrics=["mse"])
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, sample_weight):
    adam = keras.optimizers.Adam(learning_rate=0.001)
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam, loss=bce, metrics=["mse"])
    model.evaluate((x, y), sample_weight=sample_weight)


def test_model_export(model):
    out = model((x, y))
    keras.models.save_model(model.export_model, export_dir, save_format="tf")
    model_reloaded = keras.models.load_model(export_dir)
    in_reloaded = tf.concat((x, y), axis=-1)
    out_reloaded = model_reloaded(in_reloaded)
    comparison = out.numpy() == out_reloaded.numpy()
    assert comparison.all()
