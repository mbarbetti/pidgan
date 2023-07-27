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
def scheduler(staircase=False):
    from pidgan.callbacks.schedulers import LearnRateExpDecay

    sched = LearnRateExpDecay(
        optimizer=adam,
        decay_rate=0.9,
        decay_steps=1000,
        staircase=staircase,
        min_learning_rate=0.001,
        verbose=False,
        key="lr",
    )
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from pidgan.callbacks.schedulers import LearnRateExpDecay

    assert isinstance(scheduler, LearnRateExpDecay)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, keras.optimizers.Optimizer)
    assert isinstance(scheduler.decay_rate, float)
    assert isinstance(scheduler.decay_steps, int)
    assert isinstance(scheduler.staircase, bool)
    assert isinstance(scheduler.min_learning_rate, float)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


@pytest.mark.parametrize("staircase", [False, True])
@pytest.mark.parametrize("min_learning_rate", [None, 0.0005])
def test_sched_use(staircase, min_learning_rate):
    from pidgan.callbacks.schedulers import LearnRateExpDecay

    sched = LearnRateExpDecay(
        optimizer=adam,
        decay_rate=0.1,
        decay_steps=100,
        staircase=staircase,
        min_learning_rate=min_learning_rate,
        verbose=True,
    )
    model.compile(optimizer=adam, loss=mse)
    history = model.fit(X, Y, batch_size=500, epochs=5, callbacks=[sched])
    last_lr = float(f"{history.history['lr'][-1]:.4f}")
    if min_learning_rate is not None:
        assert last_lr == 0.0005
    else:
        assert last_lr == 0.0001
