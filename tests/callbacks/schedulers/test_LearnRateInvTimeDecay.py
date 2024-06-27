import pytest
import keras as k
import numpy as np

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
EPOCHS = 5
LEARN_RATE = 0.001
MIN_LEARN_RATE = 0.0005
ALPHA = 0.1

X = np.c_[
    np.random.uniform(-1, 1, size=CHUNK_SIZE),
    np.random.normal(0, 1, size=CHUNK_SIZE),
    np.random.exponential(5, size=CHUNK_SIZE),
]
Y = np.tanh(X[:, 0]) + 2 * X[:, 1] * X[:, 2]

model = k.Sequential()
try:
    model.add(k.layers.InputLayer(shape=(3,)))
except ValueError:
    model.add(k.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
    model.add(k.layers.Dense(units, activation="relu"))
model.add(k.layers.Dense(1))


@pytest.fixture
def scheduler(staircase=False):
    from pidgan.callbacks.schedulers import LearnRateInvTimeDecay

    adam = k.optimizers.Adam(learning_rate=LEARN_RATE)
    sched = LearnRateInvTimeDecay(
        optimizer=adam,
        decay_rate=1/ALPHA - 1,
        decay_steps=CHUNK_SIZE / BATCH_SIZE * EPOCHS,
        staircase=staircase,
        min_learning_rate=LEARN_RATE,
        verbose=False,
        key="lr",
    )
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from pidgan.callbacks.schedulers import LearnRateInvTimeDecay

    assert isinstance(scheduler, LearnRateInvTimeDecay)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, k.optimizers.Optimizer)
    assert isinstance(scheduler.decay_rate, float)
    assert isinstance(scheduler.decay_steps, int)
    assert isinstance(scheduler.staircase, bool)
    assert isinstance(scheduler.min_learning_rate, float)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


@pytest.mark.parametrize("staircase", [False, True])
@pytest.mark.parametrize("min_learning_rate", [None, MIN_LEARN_RATE])
def test_sched_use(staircase, min_learning_rate):
    from pidgan.callbacks.schedulers import LearnRateInvTimeDecay

    adam = k.optimizers.Adam(learning_rate=LEARN_RATE)
    sched = LearnRateInvTimeDecay(
        optimizer=adam,
        decay_rate=1/ALPHA - 1,
        decay_steps=CHUNK_SIZE / BATCH_SIZE * EPOCHS,
        staircase=staircase,
        min_learning_rate=min_learning_rate,
        verbose=True,
    )
    model.compile(optimizer=adam, loss=k.losses.MeanSquaredError())
    train = model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[sched])
    last_lr = float(f"{train.history['lr'][-1]:.8f}")
    if min_learning_rate is not None:
        assert last_lr == MIN_LEARN_RATE
    else:
        assert last_lr == ALPHA * LEARN_RATE
