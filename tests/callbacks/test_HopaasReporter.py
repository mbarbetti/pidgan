import os

import hopaas_client as hpc
import numpy as np
import pytest
import tensorflow as tf
import yaml

NUM_TRIALS = 1
CHUNK_SIZE = int(1e3)

here = os.path.dirname(__file__)
tests_dir = "/".join(here.split("/")[:-1])
with open(f"{tests_dir}/config/hopaas.yml") as file:
    config = yaml.full_load(file)

client = hpc.Client(server=f"{config['server']}", token=f"{config['token']}")
properties = {"learning_rate": hpc.suggestions.Float(1e-4, 1e-3)}

X = np.c_[
    np.random.uniform(-1, 1, size=CHUNK_SIZE),
    np.random.normal(0, 1, size=CHUNK_SIZE),
    np.random.exponential(5, size=CHUNK_SIZE),
]
Y = np.tanh(X[:, 0]) + 2 * X[:, 1] * X[:, 2]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
for units in [16, 16, 16]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
model.add(tf.keras.layers.Dense(1))


@pytest.fixture
def callback():
    study = hpc.Study(
        name="Test::PIDGAN::HopaasReporter(cfg)",
        properties=properties,
        direction="minimize",
        pruner=hpc.pruners.MedianPruner(),
        sampler=hpc.samplers.TPESampler(),
        client=client,
    )
    from pidgan.callbacks import HopaasReporter

    with study.trial() as trial:
        report = HopaasReporter(
            trial=trial, loss_name="loss", report_frequency=1, enable_pruning=False
        )
        trial.loss = 42
    return report


###########################################################################


def test_callback_configuration(callback):
    from pidgan.callbacks import HopaasReporter

    assert isinstance(callback, HopaasReporter)
    assert isinstance(callback.loss_name, str)
    assert isinstance(callback.report_frequency, int)
    assert isinstance(callback.enable_pruning, bool)


@pytest.mark.parametrize("enable_pruning", [False, True])
def test_callback_use(enable_pruning):
    study = hpc.Study(
        name="Test::PIDGAN::HopaasReporter(use)",
        properties=properties,
        direction="minimize",
        pruner=hpc.pruners.MedianPruner(),
        sampler=hpc.samplers.TPESampler(),
        client=client,
    )
    from pidgan.callbacks import HopaasReporter

    for _ in range(NUM_TRIALS):
        with study.trial() as trial:
            adam = tf.keras.optimizers.Adam(learning_rate=trial.learning_rate)
            mse = tf.keras.losses.MeanSquaredError()

            report = HopaasReporter(
                trial=trial,
                loss_name="loss",
                report_frequency=1,
                enable_pruning=enable_pruning,
            )

            model.compile(optimizer=adam, loss=mse)
            model.fit(X, Y, batch_size=100, epochs=5, callbacks=[report])
