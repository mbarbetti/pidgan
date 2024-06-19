import pytest
import keras as k
import numpy as np

CHUNK_SIZE = int(1e4)
LEARN_RATE = 0.001

X = np.c_[
    np.random.uniform(-1, 1, size=CHUNK_SIZE),
    np.random.normal(0, 1, size=CHUNK_SIZE),
    np.random.exponential(5, size=CHUNK_SIZE),
]
Y = np.tanh(X[:, 0]) + 2 * X[:, 1] * X[:, 2]

model = k.Sequential()
try:
    model.add(k.layers.InputLayer(shape=(3,)))
except(ValueError):
    model.add(k.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
    model.add(k.layers.Dense(units, activation="relu"))
model.add(k.layers.Dense(1))


@pytest.fixture
def scheduler():
    from pidgan.callbacks.schedulers import LearnRateBaseScheduler

    adam = k.optimizers.Adam(learning_rate=LEARN_RATE)
    sched = LearnRateBaseScheduler(optimizer=adam, verbose=True, key="lr")
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from pidgan.callbacks.schedulers import LearnRateBaseScheduler

    assert isinstance(scheduler, LearnRateBaseScheduler)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, k.optimizers.Optimizer)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


def test_sched_use(scheduler):
    model.compile(optimizer=scheduler.optimizer, loss=k.losses.MeanSquaredError())
    history = model.fit(X, Y, batch_size=512, epochs=10, callbacks=[scheduler])
    last_lr = float(f"{history.history['lr'][-1]:.3f}")
    assert last_lr == LEARN_RATE
    #raise TypeError
