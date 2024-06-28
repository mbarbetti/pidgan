from pidgan.algorithms.k3.lipschitz_regularizations import (
    PENALTY_STRATEGIES,
    compute_GradientPenalty,
)
from pidgan.algorithms.k3.WGAN import WGAN

LIPSCHITZ_CONSTANT = 1.0


class WGAN_GP(WGAN):
    def __init__(
        self,
        generator,
        discriminator,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="two-sided",
        feature_matching_penalty=0.0,
        referee=None,
        name="WGAN-GP",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )
        self._clip_param = None

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

    def _tf_d_train_step(self, x, y, sample_weight=None) -> None:
        super(WGAN, self)._tf_d_train_step(x, y, sample_weight)

    def _compute_d_loss(self, x, y, sample_weight=None, training=True, test=False):
        d_loss = super()._compute_d_loss(x, y, sample_weight, training)
        if not test:
            d_loss += self._lipschitz_regularization(
                self._discriminator,
                x,
                y,
                sample_weight,
                training_discriminator=training,
            )
        return d_loss

    def _lipschitz_regularization(
        self, discriminator, x, y, sample_weight=None, training_discriminator=True
    ):
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
