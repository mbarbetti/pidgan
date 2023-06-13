import numpy as np
import pytest

CHUNK_SIZE = int(1e4)

y_true = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
y_pred = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
weight = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE,))


@pytest.fixture
def metric():
    from pidgan.metrics import KLDivergence

    metric_ = KLDivergence()
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from pidgan.metrics import KLDivergence

    assert isinstance(metric, KLDivergence)
    assert isinstance(metric.name, str)


def test_metric_use_no_weights(metric):
    metric.update_state(y_true, y_pred, sample_weight=None)
    res = metric.result().numpy()
    assert res


def test_metric_use_with_weights(metric):
    metric.update_state(y_true, y_pred, sample_weight=weight)
    res = metric.result().numpy()
    assert res
