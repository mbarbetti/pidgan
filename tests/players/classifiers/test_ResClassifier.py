import os

import pytest
import keras as k
import tensorflow as tf

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/res-classifier"

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))
labels = tf.random.uniform(shape=(CHUNK_SIZE,), minval=0.0, maxval=1.0)
labels = tf.cast(labels > 0.5, x.dtype)


@pytest.fixture
def model():
    from pidgan.players.classifiers import ResClassifier

    clf = ResClassifier(
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_hidden_activation="relu",
        mlp_hidden_kernel_regularizer="l2",
        mlp_dropout_rates=0.0,
    )
    return clf


###########################################################################


def test_model_configuration(model):
    from pidgan.players.classifiers import ResClassifier

    assert isinstance(model, ResClassifier)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, int)
    # assert isinstance(model.mlp_hidden_activation, str)
    # assert isinstance(model.mlp_hidden_kernel_regularizer, str)
    assert isinstance(model.mlp_dropout_rates, float)


@pytest.mark.parametrize("mlp_hidden_activation", ["relu", "leaky_relu"])
@pytest.mark.parametrize("inputs", [y, (x, y)])
def test_model_use(mlp_hidden_activation, inputs):
    from pidgan.players.classifiers import ResClassifier

    model = ResClassifier(
        num_hidden_layers=3,
        mlp_hidden_units=128,
        mlp_hidden_activation=mlp_hidden_activation,
        mlp_dropout_rates=0.0,
    )

    out = model(inputs)
    model.summary()
    test_shape = [x.shape[0], 1]
    assert out.shape == tuple(test_shape)
    assert isinstance(model.plain_keras, k.Model)


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
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=0.001),
        loss=k.losses.MeanSquaredError(),
        metrics=["mae"],
    )
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("inputs", [y, (x, y)])
@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, inputs, sample_weight):
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=0.001),
        loss=k.losses.MeanSquaredError(),
        metrics=["mae"],
    )
    model.evaluate(x=inputs, y=labels, sample_weight=sample_weight)


@pytest.mark.parametrize("inputs", [y[:BATCH_SIZE], (x[:BATCH_SIZE], y[:BATCH_SIZE])])
def test_model_export(model, inputs):
    out = model(inputs)

    v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]
    if v_major == 3 and v_minor >= 0:
        model.plain_keras.export(export_dir)
        model_reloaded = k.layers.TFSMLayer(export_dir, call_endpoint="serve")
    else:
        k.models.save_model(model.plain_keras, export_dir, save_format="tf")
        model_reloaded = k.models.load_model(export_dir)

    if isinstance(inputs, (list, tuple)):
        in_reloaded = tf.concat((x[:BATCH_SIZE], y[:BATCH_SIZE]), axis=-1)
    else:
        in_reloaded = inputs
    out_reloaded = model_reloaded(in_reloaded)
    comparison = out.numpy() == out_reloaded.numpy()
    assert comparison.all()
