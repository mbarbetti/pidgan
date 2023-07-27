import numpy as np
import pytest
from tensorflow import keras

CHUNK_SIZE = int(1e4)

X = np.c_[
    np.random.uniform(-1, 1, size=CHUNK_SIZE),
    np.random.normal(0, 1, size=CHUNK_SIZE),
    np.random.exponential(5, size=CHUNK_SIZE),
]
Y = np.tanh(X[:, 0]) + 2 * X[:, 1] * X[:, 2]

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
    model.add(keras.layers.Dense(units, activation="relu"))
model.add(keras.layers.Dense(1))

adam = keras.optimizers.Adam(learning_rate=0.001)
mse = keras.losses.MeanSquaredError()


@pytest.fixture
def scheduler():
    from pidgan.callbacks.schedulers import LearnRatePiecewiseConstDecay

    sched = LearnRatePiecewiseConstDecay(
        optimizer=adam,
        boundaries=[25, 50],
        values=[0.001, 0.0005, 0.0001],
        verbose=True,
        key="lr",
    )
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from pidgan.callbacks.schedulers import LearnRatePiecewiseConstDecay

    assert isinstance(scheduler, LearnRatePiecewiseConstDecay)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, keras.optimizers.Optimizer)
    assert isinstance(scheduler.boundaries, list)
    assert isinstance(scheduler.boundaries[0], int)
    assert isinstance(scheduler.values, list)
    assert isinstance(scheduler.values[0], float)
    assert len(scheduler.boundaries) == len(scheduler.values)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


def test_sched_use(scheduler):
    model.compile(optimizer=adam, loss=mse)
    history = model.fit(X, Y, batch_size=500, epochs=5, callbacks=[scheduler])
    last_lr = float(f"{history.history['lr'][-1]:.4f}")
    assert last_lr == 0.0001
