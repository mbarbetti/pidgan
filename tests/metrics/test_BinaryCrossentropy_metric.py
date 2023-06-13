import numpy as np
import pytest

CHUNK_SIZE = int(1e4)
y_true = None
y_pred = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE, 1))
y_pred_logits = np.random.uniform(-5.0, 5.0, size=(CHUNK_SIZE, 1))
weight = np.random.uniform(0.0, 1.0, size=(CHUNK_SIZE,))


@pytest.fixture
def metric():
    from pidgan.metrics import BinaryCrossentropy

    metric_ = BinaryCrossentropy(from_logits=False, label_smoothing=0.0)
    return metric_


###########################################################################


def test_metric_configuration(metric):
    from pidgan.metrics import BinaryCrossentropy

    assert isinstance(metric, BinaryCrossentropy)
    assert isinstance(metric.name, str)


@pytest.mark.parametrize("from_logits", [False, True])
def test_metric_use_no_weights(from_logits):
    from pidgan.metrics import BinaryCrossentropy

    metric = BinaryCrossentropy(from_logits=from_logits, label_smoothing=0.0)
    if from_logits:
        metric.update_state(y_true, y_pred_logits, sample_weight=None)
        res = metric.result().numpy()
    else:
        metric.update_state(y_true, y_pred, sample_weight=None)
        res = metric.result().numpy()
    assert res


@pytest.mark.parametrize("from_logits", [False, True])
def test_metric_use_with_weights(from_logits):
    from pidgan.metrics import BinaryCrossentropy

    metric = BinaryCrossentropy(from_logits=from_logits, label_smoothing=0.0)
    if from_logits:
        metric.update_state(y_true, y_pred_logits, sample_weight=weight)
        res = metric.result().numpy()
    else:
        metric.update_state(y_true, y_pred, sample_weight=weight)
        res = metric.result().numpy()
    assert res
