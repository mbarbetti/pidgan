import numpy as np
import pytest

CHUNK_SIZE = int(1e4)

y_true = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
y_pred = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
weight = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE,))


@pytest.fixture
def metric():
    from pidgan.metrics import WassersteinDistance

    metric_ = WassersteinDistance()
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from pidgan.metrics import WassersteinDistance

    assert isinstance(metric, WassersteinDistance)
    assert isinstance(metric.name, str)


@pytest.mark.parametrize("sample_weight", [None, weight])
def test_metric_use(metric, sample_weight):
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    res = metric.result().numpy()
    assert res
