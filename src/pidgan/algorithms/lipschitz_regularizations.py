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
) -> tf.Tensor:
    x_ref, y_ref, _ = trainset_ref
    x_gen, y_gen, _ = trainset_gen

    x_concat = tf.concat([x_ref, x_gen], axis=0)
    y_concat = tf.concat([y_ref, y_gen], axis=0)

    with tf.GradientTape() as tape:
        # Compute interpolated points
        eps = tf.random.uniform(
            shape=(tf.shape(y_ref)[0],), minval=0.0, maxval=1.0, dtype=y_ref.dtype
        )[:, None]
        x_hat = tf.clip_by_value(
            x_gen + eps * (x_ref - x_gen),
            clip_value_min=tf.reduce_min(x_concat, axis=0),
            clip_value_max=tf.reduce_max(x_concat, axis=0),
        )
        y_hat = tf.clip_by_value(
            y_gen + eps * (y_ref - y_gen),
            clip_value_min=tf.reduce_min(y_concat, axis=0),
            clip_value_max=tf.reduce_max(y_concat, axis=0),
        )
        d_in_hat = tf.concat((x_hat, y_hat), axis=-1)
        tape.watch(d_in_hat)

        # Value of the discriminator on interpolated points
        x_hat = d_in_hat[:, :tf.shape(x_hat)[1]]
        y_hat = d_in_hat[:, tf.shape(x_hat)[1]:]
        d_out_hat = discriminator((x_hat, y_hat), training=training_discriminator)
        grad = tape.gradient(d_out_hat, d_in_hat)
        norm = tf.norm(grad, axis=-1)

    if lipschitz_penalty_strategy == "two-sided":
        gp_term = (norm - lipschitz_constant) ** 2
    else:
        gp_term = (tf.maximum(0.0, norm - lipschitz_constant)) ** 2
    return lipschitz_penalty * tf.reduce_mean(gp_term)


def compute_CriticGradientPenalty(
    critic,
    trainset_ref,
    trainset_gen_1,
    trainset_gen_2,
    training_critic=True,
    lipschitz_penalty=1.0,
    lipschitz_penalty_strategy="two-sided",
    lipschitz_constant=LIPSCHITZ_CONSTANT,
) -> tf.Tensor:
    x_ref, y_ref, _ = trainset_ref
    x_gen_1, y_gen_1, _ = trainset_gen_1
    x_gen_2, y_gen_2, _ = trainset_gen_2

    x_concat = tf.concat([x_ref, x_gen_1], axis=0)
    y_concat = tf.concat([y_ref, y_gen_1], axis=0)

    with tf.GradientTape() as tape:
        # Compute interpolated points
        eps = tf.random.uniform(
            shape=(tf.shape(y_ref)[0],), minval=0.0, maxval=1.0, dtype=y_ref.dtype
        )[:, None]
        x_hat = tf.clip_by_value(
            x_gen_1 + eps * (x_ref - x_gen_1),
            clip_value_min=tf.reduce_min(x_concat, axis=0),
            clip_value_max=tf.reduce_max(x_concat, axis=0),
        )
        y_hat = tf.clip_by_value(
            y_gen_1 + eps * (y_ref - y_gen_1),
            clip_value_min=tf.reduce_min(y_concat, axis=0),
            clip_value_max=tf.reduce_max(y_concat, axis=0),
        )
        c_in_hat = tf.concat((x_hat, y_hat), axis=-1)
        tape.watch(c_in_hat)

        # Value of the critic on interpolated points
        x_hat = c_in_hat[:, :tf.shape(x_hat)[1]]
        y_hat = c_in_hat[:, tf.shape(x_hat)[1]:]
        c_out_hat = critic((x_hat, y_hat), (x_gen_2, y_gen_2), training=training_critic)
        grad = tape.gradient(c_out_hat, c_in_hat)
        norm = tf.norm(grad, axis=-1)

    if lipschitz_penalty_strategy == "two-sided":
        gp_term = (norm - lipschitz_constant) ** 2
    else:
        gp_term = (tf.maximum(0.0, norm - lipschitz_constant)) ** 2
    return lipschitz_penalty * tf.reduce_mean(gp_term)


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
) -> tf.Tensor:
    x_ref, y_ref, _ = trainset_ref
    x_gen, y_gen, _ = trainset_gen

    x_concat = tf.concat([x_ref, x_gen], axis=0)
    y_concat = tf.concat([y_ref, y_gen], axis=0)
    d_out = discriminator((x_concat, y_concat), training=training_discriminator)

    # Initial virtual adversarial direction
    adv_dir = tf.random.uniform(
        shape=tf.shape(y_concat), minval=-1.0, maxval=1.0, dtype=y_concat.dtype
    )
    adv_dir /= tf.norm(adv_dir, axis=-1, keepdims=True)

    for _ in range(vir_adv_dir_upds):
        with tf.GradientTape() as tape:
            tape.watch(adv_dir)
            xi = tf.math.reduce_std(y_concat, axis=0, keepdims=True)
            y_hat = tf.clip_by_value(
                y_concat + xi * adv_dir,
                clip_value_min=tf.reduce_min(y_concat, axis=0),
                clip_value_max=tf.reduce_max(y_concat, axis=0),
            )
            d_out_hat = discriminator(
                (x_concat, y_hat), training=training_discriminator
            )
            d_diff = tf.reduce_mean(tf.abs(d_out - d_out_hat))
            grad = tape.gradient(d_diff, adv_dir)
            adv_dir = grad / tf.maximum(
                tf.norm(grad, axis=-1, keepdims=True), MIN_STABLE_VALUE
            )

    # Virtual adversarial direction
    xi = tf.random.uniform(
        shape=(tf.shape(y_concat)[0],),
        minval=xi_min,
        maxval=xi_max,
        dtype=y_concat.dtype,
    )
    xi = tf.tile(xi[:, None], (1, tf.shape(y_concat)[1]))
    y_hat = tf.clip_by_value(
        y_concat + xi * adv_dir,
        clip_value_min=tf.reduce_min(y_concat, axis=0),
        clip_value_max=tf.reduce_max(y_concat, axis=0),
    )
    d_out_hat = discriminator((x_concat, y_hat), training=training_discriminator)

    d_diff = tf.abs(d_out - d_out_hat)
    y_diff = tf.norm(y_concat - y_hat, axis=-1, keepdims=True)
    d_diff_stable = tf.boolean_mask(d_diff, mask=y_diff > MIN_STABLE_VALUE)
    y_diff_stable = tf.boolean_mask(y_diff, mask=y_diff > MIN_STABLE_VALUE)
    K = d_diff_stable / y_diff_stable  # lipschitz constant

    if lipschitz_penalty_strategy == "two-sided":
        alp_term = tf.abs(K - lipschitz_constant)
    else:
        alp_term = tf.maximum(0.0, K - lipschitz_constant)
    return lipschitz_penalty * tf.reduce_mean(alp_term) ** 2
