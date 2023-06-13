import pytest
import tensorflow as tf

CHUNK_SIZE = int(1e4)

x = tf.random.normal(shape=(CHUNK_SIZE, 4))
y = tf.random.normal(shape=(CHUNK_SIZE, 8))
w = tf.random.uniform(shape=(CHUNK_SIZE,))


@pytest.fixture
def model():
    from pidgan.players.generators import Generator

    gen = Generator(
        output_dim=y.shape[1],
        latent_dim=64,
        num_hidden_layers=5,
        mlp_hidden_units=128,
        dropout_rate=0.0,
        output_activation=None,
    )
    return gen


###########################################################################


def test_model_configuration(model):
    from pidgan.players.generators import Generator

    assert isinstance(model, Generator)
    assert isinstance(model.output_dim, int)
    assert isinstance(model.latent_dim, int)
    assert isinstance(model.num_hidden_layers, int)
    assert isinstance(model.mlp_hidden_units, list)
    assert isinstance(model.dropout_rate, list)


def test_model_use(model):
    output = model(x)
    model.summary()
    test_shape = [x.shape[0]]
    test_shape.append(model.output_dim)
    assert output.shape == tuple(test_shape)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_train(model, sample_weight):
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
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse)
    model.fit(dataset, epochs=2)


@pytest.mark.parametrize("sample_weight", [w, None])
def test_model_eval(model, sample_weight):
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=adam, loss=mse, metrics=["mse"])
    model.evaluate(x, sample_weight=sample_weight)
