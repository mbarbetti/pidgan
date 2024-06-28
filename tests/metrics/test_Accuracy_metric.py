import numpy as np
import pytest

CHUNK_SIZE = int(1e4)

y_true = None
y_pred = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
weight = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE,))


@pytest.fixture
def metric():
    from pidgan.metrics import Accuracy

    metric_ = Accuracy(threshold=0.5)
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from pidgan.metrics import Accuracy

    assert isinstance(metric, Accuracy)
    assert isinstance(metric.name, str)


@pytest.mark.parametrize("sample_weight", [None, weight])
def test_metric_use(metric, sample_weight):
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    res = metric.result().numpy()
    assert res
