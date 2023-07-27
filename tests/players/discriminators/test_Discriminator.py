import pytest
import tensorflow as tf
from tensorflow import keras

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))
labels = tf.random.uniform(shape=(CHUNK_SIZE,), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, x.dtype)


@pytest.fixture
def model():
    from pidgan.players.discriminators import Discriminator

    disc = Discriminator(
        output_dim=1,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        dropout_rate=0.0,
        output_activation="sigmoid",
    )
    return disc


###########################################################################


def test_model_configuration(model):
    from pidgan.players.discriminators import Discriminator

    assert isinstance(model, Discriminator)
    assert isinstance(model.output_dim, int)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, list)
    assert isinstance(model.dropout_rate, list)
    # assert isinstance(model.output_activation, str)
    assert isinstance(model.export_model, keras.Sequential)


def test_model_use(model):
    out = model((x, y))
    model.summary()
    test_shape = [x.shape[0]]
    test_shape.append(model.output_dim)
    assert out.shape == tuple(test_shape)
    hidden_feat, hidden_idx = model.hidden_feature((x, y), return_hidden_idx=True)
    test_shape = [x.shape[0]]
    test_shape.append(model.mlp_hidden_units[hidden_idx])
    assert hidden_feat.shape == tuple(test_shape)


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
    model.compile(optimizer=adam, loss=bce, metrics=["bce"])
    model.fit(dataset, epochs=1)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, sample_weight):
    adam = keras.optimizers.Adam(learning_rate=0.001)
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam, loss=bce, metrics=["bce"])
    model.evaluate((x, y), sample_weight=sample_weight)
