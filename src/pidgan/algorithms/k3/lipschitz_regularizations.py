import keras as k
import tensorflow as tf

PENALTY_STRATEGIES = ["two-sided", "one-sided"]
LIPSCHITZ_CONSTANT = 1.0
MIN_STABLE_VALUE = 1e-6


def compute_GradientPenalty(
    discriminator,
    trainset_ref,
    trainset_gen,
    training_discriminator=True,
    lipschitz_penalty=1.0,
    lipschitz_penalty_strategy="two-sided",
    lipschitz_constant=LIPSCHITZ_CONSTANT,
):
    x_ref, y_ref, _ = trainset_ref
    x_gen, y_gen, _ = trainset_gen

    x_concat = k.ops.concatenate([x_ref, x_gen], axis=0)
    y_concat = k.ops.concatenate([y_ref, y_gen], axis=0)

    if k.backend.backend() == "tensorflow":
        with tf.GradientTape() as tape:
            # Compute interpolated points
            eps = k.random.uniform(
                shape=(k.ops.shape(y_ref)[0],),
                minval=0.0,
                maxval=1.0,
                dtype=y_ref.dtype,
            )[:, None]
            x_hat = k.ops.clip(
                x_gen + eps * (x_ref - x_gen),
                x_min=k.ops.min(x_concat, axis=0),
                x_max=k.ops.max(x_concat, axis=0),
            )
            y_hat = k.ops.clip(
                y_gen + eps * (y_ref - y_gen),
                x_min=k.ops.min(y_concat, axis=0),
                x_max=k.ops.max(y_concat, axis=0),
            )
            d_in_hat = k.ops.concatenate((x_hat, y_hat), axis=-1)
            tape.watch(d_in_hat)

            # Value of the discriminator on interpolated points
            x_hat = d_in_hat[:, : k.ops.shape(x_hat)[1]]
            y_hat = d_in_hat[:, k.ops.shape(x_hat)[1] :]
            d_out_hat = discriminator((x_hat, y_hat), training=training_discriminator)
            grad = tape.gradient(d_out_hat, d_in_hat)
            norm = k.ops.norm(grad, axis=-1)

    elif k.backend.backend() == "torch":
        raise NotImplementedError(
            '"compute_GradientPenalty()" not implemented for the PyTorch backend'
        )
    elif k.backend.backend() == "jax":
        raise NotImplementedError(
            '"compute_GradientPenalty()" not implemented for the Jax backend'
        )

    if lipschitz_penalty_strategy == "two-sided":
        gp_term = (norm - lipschitz_constant) ** 2
    else:
        gp_term = (k.ops.maximum(0.0, norm - lipschitz_constant)) ** 2
    return lipschitz_penalty * k.ops.mean(gp_term)


def compute_CriticGradientPenalty(
    critic,
    trainset_ref,
    trainset_gen_1,
    trainset_gen_2,
    training_critic=True,
    lipschitz_penalty=1.0,
    lipschitz_penalty_strategy="two-sided",
    lipschitz_constant=LIPSCHITZ_CONSTANT,
):
    x_ref, y_ref, _ = trainset_ref
    x_gen_1, y_gen_1, _ = trainset_gen_1
    x_gen_2, y_gen_2, _ = trainset_gen_2

    x_concat = k.ops.concatenate([x_ref, x_gen_1], axis=0)
    y_concat = k.ops.concatenate([y_ref, y_gen_1], axis=0)

    if k.backend.backend() == "tensorflow":
        with tf.GradientTape() as tape:
            # Compute interpolated points
            eps = k.random.uniform(
                shape=(k.ops.shape(y_ref)[0],),
                minval=0.0,
                maxval=1.0,
                dtype=y_ref.dtype,
            )[:, None]
            x_hat = k.ops.clip(
                x_gen_1 + eps * (x_ref - x_gen_1),
                x_min=k.ops.min(x_concat, axis=0),
                x_max=k.ops.max(x_concat, axis=0),
            )
            y_hat = k.ops.clip(
                y_gen_1 + eps * (y_ref - y_gen_1),
                x_min=k.ops.min(y_concat, axis=0),
                x_max=k.ops.max(y_concat, axis=0),
            )
            c_in_hat = k.ops.concatenate((x_hat, y_hat), axis=-1)
            tape.watch(c_in_hat)

            # Value of the critic on interpolated points
            x_hat = c_in_hat[:, : k.ops.shape(x_hat)[1]]
            y_hat = c_in_hat[:, k.ops.shape(x_hat)[1] :]
            c_out_hat = critic(
                (x_hat, y_hat), (x_gen_2, y_gen_2), training=training_critic
            )
            grad = tape.gradient(c_out_hat, c_in_hat)
            norm = k.ops.norm(grad, axis=-1)

    elif k.backend.backend() == "torch":
        raise NotImplementedError(
            '"compute_CriticGradientPenalty()" not implemented for the PyTorch backend'
        )
    elif k.backend.backend() == "jax":
        raise NotImplementedError(
            '"compute_CriticGradientPenalty()" not implemented for the Jax backend'
        )

    if lipschitz_penalty_strategy == "two-sided":
        gp_term = (norm - lipschitz_constant) ** 2
    else:
        gp_term = (k.ops.maximum(0.0, norm - lipschitz_constant)) ** 2
    return lipschitz_penalty * k.ops.mean(gp_term)


def compute_AdversarialLipschitzPenalty(
    discriminator,
    trainset_ref,
    trainset_gen,
    training_discriminator=True,
    vir_adv_dir_upds=1,
    xi_min=0.1,
    xi_max=10.0,
    lipschitz_penalty=1.0,
    lipschitz_penalty_strategy="one-sided",
    lipschitz_constant=LIPSCHITZ_CONSTANT,
):
    x_ref, y_ref, _ = trainset_ref
    x_gen, y_gen, _ = trainset_gen

    x_concat = k.ops.concatenate([x_ref, x_gen], axis=0)
    y_concat = k.ops.concatenate([y_ref, y_gen], axis=0)
    d_out = discriminator((x_concat, y_concat), training=training_discriminator)

    # Initial virtual adversarial direction
    adv_dir = k.random.uniform(
        shape=k.ops.shape(y_concat), minval=-1.0, maxval=1.0, dtype=y_concat.dtype
    )
    adv_dir /= k.ops.norm(adv_dir, axis=-1, keepdims=True)

    if k.backend.backend() == "tensorflow":
        for _ in range(vir_adv_dir_upds):
            with tf.GradientTape() as tape:
                tape.watch(adv_dir)
                xi = k.ops.std(y_concat, axis=0, keepdims=True)
                y_hat = k.ops.clip(
                    y_concat + xi * adv_dir,
                    x_min=k.ops.min(y_concat, axis=0),
                    x_max=k.ops.max(y_concat, axis=0),
                )
                d_out_hat = discriminator(
                    (x_concat, y_hat), training=training_discriminator
                )
                d_diff = k.ops.mean(k.ops.abs(d_out - d_out_hat))
                grad = tape.gradient(d_diff, adv_dir)
                adv_dir = grad / k.ops.maximum(
                    k.ops.norm(grad, axis=-1, keepdims=True), MIN_STABLE_VALUE
                )

    elif k.backend.backend() == "torch":
        raise NotImplementedError(
            '"compute_AdversarialLipschitzPenalty()" not '
            "implemented for the PyTorch backend"
        )
    elif k.backend.backend() == "jax":
        raise NotImplementedError(
            '"compute_AdversarialLipschitzPenalty()" not '
            "implemented for the Jax backend"
        )

    # Virtual adversarial direction
    xi = k.random.uniform(
        shape=(k.ops.shape(y_concat)[0],),
        minval=xi_min,
        maxval=xi_max,
        dtype=y_concat.dtype,
    )
    xi = k.ops.tile(xi[:, None], (1, k.ops.shape(y_concat)[1]))
    y_hat = k.ops.clip(
        y_concat + xi * adv_dir,
        x_min=k.ops.min(y_concat, axis=0),
        x_max=k.ops.max(y_concat, axis=0),
    )
    d_out_hat = discriminator((x_concat, y_hat), training=training_discriminator)

    d_diff = k.ops.abs(d_out - d_out_hat)
    y_diff = k.ops.norm(y_concat - y_hat, axis=-1, keepdims=True)
    d_diff_stable = d_diff[y_diff > MIN_STABLE_VALUE]
    y_diff_stable = y_diff[y_diff > MIN_STABLE_VALUE]
    K = d_diff_stable / y_diff_stable  # lipschitz constant

    if lipschitz_penalty_strategy == "two-sided":
        alp_term = k.ops.abs(K - lipschitz_constant)
    else:
        alp_term = k.ops.maximum(0.0, K - lipschitz_constant)
    return lipschitz_penalty * k.ops.mean(alp_term) ** 2
