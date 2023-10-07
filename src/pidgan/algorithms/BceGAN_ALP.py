import tensorflow as tf

from pidgan.algorithms.BceGAN_GP import BceGAN_GP
from pidgan.algorithms.lipschitz_regularizations import (
    compute_AdversarialLipschitzPenalty,
)

LIPSCHITZ_CONSTANT = 1.0
XI_MIN = 0.8
XI_MAX = 1.2


class BceGAN_ALP(BceGAN_GP):
    def __init__(
        self,
        generator,
        discriminator,
        lipschitz_penalty=1.0,
        lipschitz_penalty_strategy="one-sided",
        feature_matching_penalty=0.0,
        referee=None,
        name="BceGAN-ALP",
        dtype=None,
    ):
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=lipschitz_penalty,
            lipschitz_penalty_strategy=lipschitz_penalty_strategy,
            feature_matching_penalty=feature_matching_penalty,
            referee=referee,
            name=name,
            dtype=dtype,
        )

    def compile(
        self,
        metrics=None,
        generator_optimizer="rmsprop",
        discriminator_optimizer="rmsprop",
        generator_upds_per_batch=1,
        discriminator_upds_per_batch=1,
        virtual_adv_direction_upds=1,
        referee_optimizer=None,
        referee_upds_per_batch=None,
    ) -> None:
        super().compile(
            metrics=metrics,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_upds_per_batch=generator_upds_per_batch,
            discriminator_upds_per_batch=discriminator_upds_per_batch,
            referee_optimizer=referee_optimizer,
            referee_upds_per_batch=referee_upds_per_batch,
        )

        # Virtual adversarial direction updates
        assert isinstance(virtual_adv_direction_upds, (int, float))
        assert virtual_adv_direction_upds > 0
        self._vir_adv_dir_upds = int(virtual_adv_direction_upds)

    def _lipschitz_regularization(
        self, discriminator, x, y, sample_weight=None, training_discriminator=True
    ) -> tf.Tensor:
        trainset_ref, trainset_gen = self._prepare_trainset(
            x, y, sample_weight, training_generator=False
        )
        return compute_AdversarialLipschitzPenalty(
            discriminator=discriminator,
            trainset_ref=trainset_ref,
            trainset_gen=trainset_gen,
            training_discriminator=training_discriminator,
            vir_adv_dir_upds=self._vir_adv_dir_upds,
            xi_min=XI_MIN,
            xi_max=XI_MAX,
            lipschitz_penalty=self._lipschitz_penalty,
            lipschitz_penalty_strategy=self._lipschitz_penalty_strategy,
            lipschitz_constant=LIPSCHITZ_CONSTANT,
        )

    @property
    def virtual_adv_direction_upds(self) -> int:
        return self._vir_adv_dir_upds
