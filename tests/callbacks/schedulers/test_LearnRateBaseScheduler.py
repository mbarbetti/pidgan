import pytest
import keras as k
import numpy as np

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
EPOCHS = 5
LEARN_RATE = 0.001

x = np.random.normal(size=(CHUNK_SIZE, 4)).astype("float32")
y = np.random.normal(size=(CHUNK_SIZE, 1)).astype("float32")

model = k.Sequential()
try:
    model.add(k.layers.InputLayer(shape=(x.shape[1],)))
except ValueError:
    model.add(k.layers.InputLayer(input_shape=(x.shape[1],)))
for units in [16, 16, 16]:
    model.add(k.layers.Dense(units, activation="relu"))
model.add(k.layers.Dense(y.shape[1]))


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
    train = model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[scheduler])
    last_lr = float(f"{train.history['lr'][-1]:.8f}")
    assert last_lr == LEARN_RATE
