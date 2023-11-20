import os

import pytest
import tensorflow as tf
from tensorflow import keras

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/classifier"

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))
labels = tf.random.uniform(shape=(CHUNK_SIZE,), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, x.dtype)


@pytest.fixture
def model():
    from pidgan.players.classifiers import Classifier

    clf = Classifier(
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_hidden_activation="relu",
        mlp_dropout_rates=0.0,
    )
    return clf


###########################################################################


def test_model_configuration(model):
    from pidgan.players.classifiers import Classifier

    assert isinstance(model, Classifier)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, list)
    # assert isinstance(model.mlp_hidden_activation, str)
    assert isinstance(model.mlp_dropout_rates, list)


@pytest.mark.parametrize("mlp_hidden_units", [128, [128, 128, 128]])
@pytest.mark.parametrize("mlp_hidden_activation", ["relu", "leaky_relu"])
@pytest.mark.parametrize("mlp_dropout_rates", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("inputs", [y, (x, y)])
def test_model_use(mlp_hidden_units, mlp_hidden_activation, mlp_dropout_rates, inputs):
    from pidgan.players.classifiers import Classifier

    model = Classifier(
        num_hidden_layers=3,
        mlp_hidden_units=mlp_hidden_units,
        mlp_hidden_activation=mlp_hidden_activation,
        mlp_dropout_rates=mlp_dropout_rates,
    )
    out = model(inputs)
    model.summary()
    test_shape = [x.shape[0], 1]
    assert out.shape == tuple(test_shape)
    assert isinstance(model.export_model, keras.Sequential)


@pytest.mark.parametrize("inputs", [y, (x, y)])
@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_train(model, inputs, sample_weight):
    if sample_weight is not None:
        slices = (inputs, labels, w)
    else:
        slices = (inputs, labels)
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


@pytest.mark.parametrize("inputs", [y, (x, y)])
@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, inputs, sample_weight):
    adam = keras.optimizers.Adam(learning_rate=0.001)
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=adam, loss=bce, metrics=["mse"])
    model.evaluate(inputs, sample_weight=sample_weight)


@pytest.mark.parametrize("inputs", [y, (x, y)])
def test_model_export(model, inputs):
    out = model(inputs)
    keras.models.save_model(model.export_model, export_dir, save_format="tf")
    model_reloaded = keras.models.load_model(export_dir)
    if isinstance(inputs, (list, tuple)):
        in_reloaded = tf.concat((x, y), axis=-1)
    else:
        in_reloaded = inputs
    out_reloaded = model_reloaded(in_reloaded)
    comparison = out.numpy() == out_reloaded.numpy()
    assert comparison.all()
