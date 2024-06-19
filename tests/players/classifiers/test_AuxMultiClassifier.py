import os

import numpy as np
import pytest
import keras as k
import tensorflow as tf

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500

here = os.path.dirname(__file__)
export_dir = f"{here}/tmp/aux-multi-classifier"

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))
labels = np.random.choice(3, size=(CHUNK_SIZE,), p=[0.4, 0.2, 0.4])
labels = tf.one_hot(labels, depth=3, on_value=1.0, off_value=0.0)


@pytest.fixture
def model():
    from pidgan.players.classifiers import AuxMultiClassifier

    clf = AuxMultiClassifier(
        num_multiclasses=labels.shape[1],
        aux_features=["0 + 1", "2 - 3", "4 * 5", "6 / 7"],
        num_hidden_layers=5,
        mlp_hidden_units=128,
        mlp_hidden_activation="relu",
        mlp_hidden_kernel_regularizer="l2",
        mlp_dropout_rates=0.0,
        enable_residual_blocks=False,
    )
    return clf


###########################################################################


def test_model_configuration(model):
    from pidgan.players.classifiers import AuxMultiClassifier

    assert isinstance(model, AuxMultiClassifier)
    assert isinstance(model.aux_features, list)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, int)
    # assert isinstance(model.mlp_hidden_activation, str)
    # assert isinstance(model.mlp_hidden_kernel_regularizer, str)
    assert isinstance(model.mlp_dropout_rates, float)
    assert isinstance(model.enable_residual_blocks, bool)


@pytest.mark.parametrize("mlp_hidden_activation", ["relu", "leaky_relu"])
@pytest.mark.parametrize("enable_res_blocks", [True, False])
@pytest.mark.parametrize("inputs", [y, (x, y)])
def test_model_use(mlp_hidden_activation, enable_res_blocks, inputs):
    from pidgan.players.classifiers import AuxMultiClassifier

    model = AuxMultiClassifier(
        num_multiclasses=labels.shape[1],
        aux_features=["0 + 1", "2 - 3", "4 * 5", "6 / 7"],
        num_hidden_layers=3,
        mlp_hidden_units=128,
        mlp_hidden_activation=mlp_hidden_activation,
        mlp_dropout_rates=0.0,
        enable_residual_blocks=enable_res_blocks,
    )
    if not isinstance(inputs, (tuple, list)):
        in_shape = inputs.shape
    else:
        in_shape = (inputs[0].shape, inputs[1].shape)
    model.build(input_shape=in_shape)

    out = model(inputs)
    model.summary()
    test_shape = [x.shape[0], model.num_multiclasses]
    assert out.shape == tuple(test_shape)
    assert isinstance(model.export_model, k.Model)


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
    if not isinstance(inputs, (tuple, list)):
        in_shape = inputs.shape
    else:
        in_shape = (inputs[0].shape, inputs[1].shape)
    model.build(input_shape=in_shape)
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=0.001),
        loss=k.losses.MeanSquaredError(), 
        metrics=["mae"],
    )
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("inputs", [y, (x, y)])
@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, inputs, sample_weight):
    if not isinstance(inputs, (tuple, list)):
        in_shape = inputs.shape
    else:
        in_shape = (inputs[0].shape, inputs[1].shape)
    model.build(input_shape=in_shape)
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=0.001),
        loss=k.losses.MeanSquaredError(), 
        metrics=["mae"],
    )
    model.evaluate(x=inputs, y=labels, sample_weight=sample_weight)


@pytest.mark.parametrize("inputs", [y, (x, y)])
def test_model_export(model, inputs):
    if not isinstance(inputs, (tuple, list)):
        in_shape = inputs.shape
    else:
        in_shape = (inputs[0].shape, inputs[1].shape)
    model.build(input_shape=in_shape)
    out, aux = model(inputs, return_aux_features=True)

    v_major, v_minor, _ = [int(v) for v in k.__version__.split(".")]
    if v_major == 3 and v_minor >= 0:
        model.export_model.export(export_dir)
        model_reloaded = k.layers.TFSMLayer(export_dir, call_endpoint="serve")
    else:
        k.models.save_model(model.export_model, export_dir, save_format="tf")
        model_reloaded = k.models.load_model(export_dir)

    if isinstance(inputs, (list, tuple)):
        in_reloaded = tf.concat((x, y, aux), axis=-1)
    else:
        in_reloaded = tf.concat((y, aux), axis=-1)
    out_reloaded = model_reloaded(in_reloaded)
    comparison = out.numpy() == out_reloaded.numpy()
    assert comparison.all()
