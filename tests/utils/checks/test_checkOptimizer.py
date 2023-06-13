import pytest
from tensorflow.keras.optimizers import Optimizer

from pidgan.utils.checks.checkOptimizer import OPT_SHORTCUTS, TF_OPTIMIZERS

###########################################################################


@pytest.mark.parametrize("optimizer", OPT_SHORTCUTS)
def test_checker_use_strings(optimizer):
    from pidgan.utils.checks import checkOptimizer

    res = checkOptimizer(optimizer)
    assert isinstance(res, Optimizer)


@pytest.mark.parametrize("optimizer", TF_OPTIMIZERS)
def test_checker_use_classes(optimizer):
    from pidgan.utils.checks import checkOptimizer

    res = checkOptimizer(optimizer)
    assert isinstance(res, Optimizer)
