import os

import hopaas_client as hpc
import keras as k
import numpy as np
import pytest
import yaml

NUM_TRIALS = 1
CHUNK_SIZE = int(1e3)

here = os.path.dirname(__file__)
tests_dir = "/".join(here.split("/")[:-2])
with open(f"{tests_dir}/config/hopaas.yml") as file:
    config = yaml.full_load(file)

client = hpc.Client(server=f"{config['server']}", token=f"{config['token']}")
properties = {"learning_rate": hpc.suggestions.Float(1e-4, 1e-3)}

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
def callback():
    study = hpc.Study(
        name="Test::PIDGAN::HopaasPruner(cfg)",
        properties=properties,
        direction="minimize",
        pruner=hpc.pruners.MedianPruner(),
        sampler=hpc.samplers.TPESampler(),
        client=client,
    )
    from pidgan.optimization.callbacks import HopaasPruner

    with study.trial() as trial:
        report = HopaasPruner(
            trial=trial, loss_name="loss", report_frequency=1, enable_pruning=False
        )
        trial.loss = 42
    return report


###########################################################################


@pytest.mark.xfail
def test_callback_configuration(callback):
    from pidgan.optimization.callbacks import HopaasPruner

    assert isinstance(callback, HopaasPruner)
    assert isinstance(callback.loss_name, str)
    assert isinstance(callback.report_frequency, int)
    assert isinstance(callback.enable_pruning, bool)


@pytest.mark.xfail
@pytest.mark.parametrize("enable_pruning", [False, True])
def test_callback_use(enable_pruning):
    study = hpc.Study(
        name="Test::PIDGAN::HopaasPruner(use)",
        properties=properties,
        direction="minimize",
        pruner=hpc.pruners.MedianPruner(),
        sampler=hpc.samplers.TPESampler(),
        client=client,
    )
    from pidgan.optimization.callbacks import HopaasPruner

    for _ in range(NUM_TRIALS):
        with study.trial() as trial:
            adam = k.optimizers.Adam(learning_rate=trial.learning_rate)
            mse = k.losses.MeanSquaredError()

            report = HopaasPruner(
                trial=trial,
                loss_name="loss",
                report_frequency=1,
                enable_pruning=enable_pruning,
            )

            model.compile(optimizer=adam, loss=mse)
            model.fit(x, y, batch_size=100, epochs=5, callbacks=[report])
