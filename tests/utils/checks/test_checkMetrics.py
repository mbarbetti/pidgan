import pytest

from pidgan.metrics import BaseMetric
from pidgan.utils.checks.checkMetrics import METRIC_SHORTCUTS, PIDGAN_METRICS


@pytest.fixture
def checker():
    from pidgan.utils.checks import checkMetrics

    chk = checkMetrics
    return chk


###########################################################################


def test_checker_use_None(checker):
    res = checker(None)
    assert res is None


@pytest.mark.parametrize("metrics", [[s] for s in METRIC_SHORTCUTS])
def test_checker_use_strings(metrics):
    from pidgan.utils.checks import checkMetrics

    res = checkMetrics(metrics)
    assert isinstance(res, list)
    assert len(res) == 1
    for r in res:
        assert isinstance(r, BaseMetric)


@pytest.mark.parametrize("metrics", [[c] for c in PIDGAN_METRICS])
def test_checker_use_classes(metrics):
    from pidgan.utils.checks import checkMetrics

    res = checkMetrics(metrics)
    assert isinstance(res, list)
    assert len(res) == 1
    for r in res:
        assert isinstance(r, BaseMetric)


def test_checker_use_mixture(checker):
    res = checker(METRIC_SHORTCUTS + PIDGAN_METRICS)
    assert isinstance(res, list)
    assert len(res) == len(METRIC_SHORTCUTS) + len(PIDGAN_METRICS)
    for r in res:
        assert isinstance(r, BaseMetric)
