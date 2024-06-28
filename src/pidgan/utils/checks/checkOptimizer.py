import keras as k

OPT_SHORTCUTS = ["sgd", "rmsprop", "adam"]
TF_OPTIMIZERS = [
    k.optimizers.SGD(),
    k.optimizers.RMSprop(),
    k.optimizers.Adam(),
]


def checkOptimizer(optimizer) -> k.optimizers.Optimizer:
    if isinstance(optimizer, str):
        if optimizer in OPT_SHORTCUTS:
            for opt, tf_opt in zip(OPT_SHORTCUTS, TF_OPTIMIZERS):
                if optimizer == opt:
                    return tf_opt
        else:
            raise ValueError(
                f"`optimizer` should be selected in {OPT_SHORTCUTS}, "
                f"instead '{optimizer}' passed"
            )
    elif isinstance(optimizer, k.optimizers.Optimizer):
        return optimizer
    else:
        raise TypeError(
            f"`optimizer` should be a Keras' Optimizer, "
            f"instead {type(optimizer)} passed"
        )
