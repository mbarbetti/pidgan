import numpy as np
import pytest
import tensorflow as tf

chunk_size = int(1e4)

X = np.c_[
    np.random.uniform(-1, 1, size=chunk_size),
    np.random.normal(0, 1, size=chunk_size),
    np.random.exponential(5, size=chunk_size),
]
Y = np.tanh(X[:, 0]) + 2 * X[:, 1] * X[:, 2]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
model.add(tf.keras.layers.Dense(1))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
mse = tf.keras.losses.MeanSquaredError()


@pytest.fixture
def scheduler():
    from calotron.callbacks.schedulers import CosineDecay

    sched = CosineDecay(optimizer=adam, decay_steps=1000, alpha=0.95, verbose=True)
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from calotron.callbacks.schedulers import CosineDecay

    assert isinstance(scheduler, CosineDecay)
    assert isinstance(scheduler.optimizer, tf.keras.optimizers.Optimizer)
    assert isinstance(scheduler.decay_steps, int)
    assert isinstance(scheduler.alpha, float)


def test_sched_use(scheduler):
    model.compile(optimizer=adam, loss=mse)
    history = model.fit(X, Y, batch_size=500, epochs=5, callbacks=[scheduler])
    last_lr = float(f"{history.history['lr'][-1]:.4f}")
    assert last_lr == 0.0001
