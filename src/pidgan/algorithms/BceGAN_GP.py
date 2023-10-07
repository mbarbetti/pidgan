import tensorflow as tf

from pidgan.algorithms import BceGAN
from pidgan.algorithms.lipschitz_regularizations import (
    PENALTY_STRATEGIES,
    compute_GradientPenalty,
)

LIPSCHITZ_CONSTANT = 1.0


class BceGAN_GP(BceGAN):
    def __init__(
        self,
        generator,
        discriminator,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="two-sided",
        feature_matching_penalty=0.0,
        referee=None,
        name="BceGAN-GP",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            from_logits=True,
            label_smoothing=0.0,
            injected_noise_stddev=0.0,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )

        # Lipschitz penalty
        assert isinstance(lipschitz_penalty, (int, float))
        assert lipschitz_penalty > 0.0
        self._lipschitz_penalty = float(lipschitz_penalty)

        # Penalty strategy
        assert isinstance(lipschitz_penalty_strategy, str)
        if lipschitz_penalty_strategy not in PENALTY_STRATEGIES:
            raise ValueError(
                "`lipschitz_penalty_strategy` should be selected "
                f"in {PENALTY_STRATEGIES}, instead "
                f"'{lipschitz_penalty_strategy}' passed"
            )
        self._lipschitz_penalty_strategy = lipschitz_penalty_strategy

    def _compute_d_loss(self, x, y, sample_weight=None, training=True) -> tf.Tensor:
        d_loss = super()._compute_d_loss(x, y, sample_weight, training)
        d_loss += self._lipschitz_regularization(
            self._discriminator, x, y, sample_weight, training_discriminator=training
        )
        return d_loss

    def _lipschitz_regularization(
        self, discriminator, x, y, sample_weight=None, training_discriminator=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return compute_GradientPenalty(
            discriminator=discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            training_discriminator=training_discriminator,
            lipschitz_penalty=self._lipschitz_penalty,
            lipschitz_penalty_strategy=self._lipschitz_penalty_strategy,
            lipschitz_constant=LIPSCHITZ_CONSTANT,
        )

    @property
    def lipschitz_penalty(self) -> float:
        return self._lipschitz_penalty

    @property
    def lipschitz_penalty_strategy(self) -> str:
        return self._lipschitz_penalty_strategy
