import pytest
import keras as k
import numpy as np

CHUNK_SIZE = int(1e4)
BATCH_SIZE = 500
EPOCHS = 5
LEARN_RATE = 0.001
ALPHA = 0.1

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
def scheduler(cycle=False):
    from pidgan.callbacks.schedulers import LearnRatePolynomialDecay

    adam = k.optimizers.Adam(learning_rate=LEARN_RATE)
    sched = LearnRatePolynomialDecay(
        optimizer=adam,
        decay_steps=CHUNK_SIZE / BATCH_SIZE * EPOCHS,
        end_learning_rate=ALPHA * LEARN_RATE,
        power=1.0,
        cycle=cycle,
        verbose=False,
        key="lr",
    )
    return sched


###########################################################################


def test_sched_configuration(scheduler):
    from pidgan.callbacks.schedulers import LearnRatePolynomialDecay

    assert isinstance(scheduler, LearnRatePolynomialDecay)
    assert isinstance(scheduler.name, str)
    assert isinstance(scheduler.optimizer, k.optimizers.Optimizer)
    assert isinstance(scheduler.decay_steps, int)
    assert isinstance(scheduler.end_learning_rate, float)
    assert isinstance(scheduler.power, float)
    assert isinstance(scheduler.cycle, bool)
    assert isinstance(scheduler.verbose, bool)
    assert isinstance(scheduler.key, str)


@pytest.mark.parametrize("cycle", [False, True])
def test_sched_use(cycle):
    from pidgan.callbacks.schedulers import LearnRatePolynomialDecay

    adam = k.optimizers.Adam(learning_rate=LEARN_RATE)
    sched = LearnRatePolynomialDecay(
        optimizer=adam,
        decay_steps=CHUNK_SIZE / BATCH_SIZE * EPOCHS,
        end_learning_rate=ALPHA * LEARN_RATE,
        power=1.0,
        cycle=cycle,
        verbose=True,
    )
    model.compile(optimizer=adam, loss=k.losses.MeanSquaredError())
    train = model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[sched])
    last_lr = float(f"{train.history['lr'][-1]:.8f}")
    assert last_lr == 0.1 * LEARN_RATE
