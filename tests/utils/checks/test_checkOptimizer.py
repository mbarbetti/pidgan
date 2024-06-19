import pytest
import keras as k

from pidgan.utils.checks.checkOptimizer import OPT_SHORTCUTS, TF_OPTIMIZERS

###########################################################################


@pytest.mark.parametrize("optimizer", OPT_SHORTCUTS)
def test_checker_use_strings(optimizer):
    from pidgan.utils.checks import checkOptimizer

    res = checkOptimizer(optimizer)
    assert isinstance(res, k.optimizers.Optimizer)


@pytest.mark.parametrize("optimizer", TF_OPTIMIZERS)
def test_checker_use_classes(optimizer):
    from pidgan.utils.checks import checkOptimizer

    res = checkOptimizer(optimizer)
    assert isinstance(res, k.optimizers.Optimizer)
