import os
import pickle
import socket
from datetime import datetime

import hopaas_client as hpc
import numpy as np
import tensorflow as tf
import yaml
from html_reports import Report
from sklearn.utils import shuffle
from utils_argparser import argparser_optimization
from utils_training import prepare_training_plots, prepare_validation_plots

from pidgan.algorithms import GAN, BceGAN, LSGAN, WGAN, WGAN_GP, CramerGAN, WGAN_ALP
from pidgan.callbacks.schedulers import LearnRateExpDecay
from pidgan.optimization.scores import KSDistance
from pidgan.players.discriminators import Discriminator
from pidgan.players.generators import Generator
from pidgan.utils.preprocessing import invertColumnTransformer
from pidgan.utils.reports import getSummaryHTML, initHPSingleton

STUDY_NAME = "PIDGAN::GanAlgo::v0"
DTYPE = np.float32
BATCHSIZE = 512
EPOCHS = 500

P_BOUNDARIES = [0, 5, 10, 25, 100]
ETA_BOUNDARIES = [1.8, 2.7, 3.5, 4.2, 5.5]
NTRACKS_BOUNDARIES = [0, 50, 150, 300, 500]

# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_optimization(description="PIDGAN optimization setup")
args = parser.parse_args()

# +---------------+
# |   GPU setup   |
# +---------------+

avail_gpus = tf.config.list_physical_devices("GPU")

if args.gpu:
    if len(avail_gpus) == 0:
        raise RuntimeError("No GPUs available for the optimization study")

# +-------------------+
# |   Initial setup   |
# +-------------------+

hp = initHPSingleton()

with open("config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
models_dir = f"{config_dir['models_dir']}"
images_dir = f"{config_dir['images_dir']}"
reports_dir = f"{config_dir['reports_dir']}"

chunk_size = int(args.chunk_size)
train_ratio = float(args.train_ratio)

# +-----------------------------+
# |    Client initialization    |
# +-----------------------------+

with open("config/hopaas.yml") as file:
    config_hopaas = yaml.full_load(file)

server = config_hopaas["server"]
token = config_hopaas["token"]

client = hpc.Client(server=server, token=token)

# +----------------------+
# |    Study creation    |
# +----------------------+

properties = {
    "algo": hpc.suggestions.Categorical(
        ["gan", "bce-gan", "lsgan", "wgan", "wgan-gp", "cramer-gan", "wgan-alp"]
    ),
    "lip_exp": hpc.suggestions.Int(-2, 2, step=1),
    "g_lr": hpc.suggestions.Float(1e-4, 1e-3),
    "d_lr": hpc.suggestions.Float(1e-4, 1e-3),
    "gupb": hpc.suggestions.Int(1, 5, step=1),
    "dupb": hpc.suggestions.Int(1, 5, step=1),
}

properties.update(
    {"train_ratio": train_ratio, "batch_size": BATCHSIZE, "epochs": EPOCHS}
)

study = hpc.Study(
    name=STUDY_NAME,
    properties=properties,
    special_properties={
        "address": socket.gethostbyname(socket.gethostname()),
        "node_name": str(args.node_name),
    },
    direction="minimize",
    pruner=hpc.pruners.NopPruner(),
    sampler=hpc.samplers.TPESampler(n_startup_trials=50),
    client=client,
)

with study.trial() as trial:
    print(f"\n{'< ' * 30} Trial n. {trial.id} {' >' * 30}\n")

    # +------------------+
    # |   Data loading   |
    # +------------------+

    if args.fullsim:
        label = "sim"
    else:
        label = "calib"

    npzfile = np.load(f"{data_dir}/pidgan-Rich-{args.particle}-{label}-dataset.npz")

    x = npzfile["x"].astype(DTYPE)[:chunk_size]
    x_vars = [str(v) for v in npzfile["x_vars"]]
    y = npzfile["y"].astype(DTYPE)[:chunk_size]
    y_vars = [str(v) for v in npzfile["y_vars"]]
    if not args.fullsim:
        if args.weights:
            w = npzfile["w"].astype(DTYPE)[:chunk_size]
            w_var = [str(v) for v in npzfile["w_var"]]
        else:
            w = None
            w_var = None
    else:
        w = None
        w_var = None

    print(f"[INFO] Input tensor - shape: {x.shape}")
    print(f"[INFO] Output tensor - shape: {y.shape}")
    if w is not None:
        print(f"[INFO] Weight tensor - shape: {w.shape}")
        x, y, w = shuffle(x, y, w)
    else:
        x, y = shuffle(x, y)

    chunk_size = x.shape[0]
    train_size = int(train_ratio * chunk_size)

    # +-------------------------+
    # |   Dataset preparation   |
    # +-------------------------+

    x_train = x[:train_size]
    y_train = y[:train_size]
    if w is not None:
        w_train = w[:train_size]
        slices = (x_train, y_train, w_train)
    else:
        w_train = None
        slices = (x_train, y_train)
    train_ds = (
        tf.data.Dataset.from_tensor_slices(slices)
        .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    if train_ratio != 1.0:
        x_val = x[train_size:]
        y_val = y[train_size:]
        if w is not None:
            w_val = w[train_size:]
            slices = (x_train, y_train, w_train)
        else:
            w_val = None
            slices = (x_train, y_train)
        val_ds = (
            tf.data.Dataset.from_tensor_slices(slices)
            .batch(BATCHSIZE, drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        x_val = x_train
        y_val = y_train
        w_val = w_train
        val_ds = None

    # +------------------------+
    # |   Model construction   |
    # +------------------------+

    generator = Generator(
        output_dim=hp.get("g_output_dim", y.shape[1]),
        latent_dim=hp.get("g_latent_dim", 64),
        num_hidden_layers=hp.get("g_num_hidden_layers", 5),
        mlp_hidden_units=hp.get("g_mlp_hidden_units", 128),
        dropout_rate=hp.get("g_dropout_rate", 0.0),
        name="generator",
        dtype=DTYPE,
    )

    d_output_dim = 1 if trial.algo != "cramer-gan" else 256
    d_output_activation = "sigmoid" if trial.algo in ["gan", "bce-gan", "lsgan"] else "linear"

    discriminator = Discriminator(
        output_dim=hp.get("d_output_dim", d_output_dim),
        num_hidden_layers=hp.get("d_num_hidden_layers", 5),
        mlp_hidden_units=hp.get("d_mlp_hidden_units", 128),
        dropout_rate=hp.get("d_dropout_rate", 0.0),
        output_activation=hp.get("d_output_activation", d_output_activation),
        name="discriminator",
        dtype=DTYPE,
    )

    lipschitz_penalty = float(f"1e{trial.lip_exp}")

    if trial.algo == "gan":
        model = GAN(
            generator=generator,
            discriminator=discriminator,
            use_original_loss=hp.get("gan_use_original_loss", True),
            injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.1),
        )
    elif trial.algo == "bce-gan":
        model = BceGAN(
            generator=generator,
            discriminator=discriminator,
            injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.1),
            from_logits=hp.get("gan_from_logits", False),
            label_smoothing=hp.get("gan_label_smoothing", 0.0),
        )
    elif trial.algo == "lsgan":
        model = LSGAN(
            generator=generator,
            discriminator=discriminator,
            minimize_pearson_chi2=hp.get("gan_minimize_pearson_chi2", False),
            injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.0),
        )
    elif trial.algo == "wgan":
        model = WGAN(
            generator=generator,
            discriminator=discriminator,
            clip_param=hp.get("gan_clip_param", 0.01),
            from_logits=hp.get("gan_from_logits", None),
            label_smoothing=hp.get("gan_label_smoothing", None),
        )
    elif trial.algo == "wgan-gp":
        model = WGAN_GP(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", lipschitz_penalty),
            penalty_strategy=hp.get("gan_penalty_strategy", "two-sided"),
            from_logits=hp.get("gan_from_logits", None),
            label_smoothing=hp.get("gan_label_smoothing", None),
        )
    elif trial.algo == "cramer-gan":
        model = CramerGAN(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", lipschitz_penalty),
            penalty_strategy=hp.get("gan_penalty_strategy", "two-sided"),
            from_logits=hp.get("gan_from_logits", None),
            label_smoothing=hp.get("gan_label_smoothing", None),
        )
    elif trial.algo == "wgan-alp":
        model = WGAN_ALP(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", lipschitz_penalty),
            penalty_strategy=hp.get("gan_penalty_strategy", "one-sided"),
            from_logits=hp.get("gan_from_logits", None),
            label_smoothing=hp.get("gan_label_smoothing", None),
        )
    hp.get("gan_name", model.name)

    output = model(x[:BATCHSIZE], y[:BATCHSIZE])
    model.summary()

    # +----------------------+
    # |   Optimizers setup   |
    # +----------------------+

    g_opt = tf.keras.optimizers.RMSprop(hp.get("g_lr0", float(trial.g_lr)))
    hp.get("g_optimizer", g_opt.name)

    d_opt = tf.keras.optimizers.RMSprop(hp.get("d_lr0", float(trial.d_lr)))
    hp.get("d_optimizer", d_opt.name)

    # +----------------------------+
    # |   Training configuration   |
    # +----------------------------+

    metrics = ["accuracy", "bce"] if trial.algo in ["gan", "bce-gan", "lsgan"] else ["wass_dist"]

    model.compile(
        metrics=hp.get("metrics", metrics),
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        generator_upds_per_batch=hp.get("generator_upds_per_batch", int(trial.gupb)),
        discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", int(trial.dupb)),
    )

    # +--------------------------+
    # |   Callbacks definition   |
    # +--------------------------+

    callbacks = list()

    g_sched = LearnRateExpDecay(
        model.generator_optimizer,
        decay_rate=hp.get("g_decay_rate", 0.10),
        decay_steps=hp.get("g_decay_steps", 100_000),
        min_learning_rate=hp.get("g_min_learning_rate", 1e-6),
        verbose=True,
        key="g_lr",
    )
    hp.get("g_sched", g_sched.name)
    callbacks.append(g_sched)

    d_sched = LearnRateExpDecay(
        model.discriminator_optimizer,
        decay_rate=hp.get("d_decay_rate", 0.10),
        decay_steps=hp.get("d_decay_steps", 100_000),
        min_learning_rate=hp.get("d_min_learning_rate", 1e-6),
        verbose=True,
        key="d_lr",
    )
    hp.get("d_sched", d_sched.name)
    callbacks.append(d_sched)

    # +------------------------+
    # |   Training procedure   |
    # +------------------------+

    start = datetime.now()
    train = model.fit(
        train_ds,
        epochs=hp.get("epochs", EPOCHS),
        validation_data=val_ds,
        callbacks=callbacks,
    )
    stop = datetime.now()

    duration = str(stop - start).split(".")[0].split(":")  # [HH, MM, SS]
    duration = f"{duration[0]}h {duration[1]}min {duration[2]}s"
    print(f"[INFO] Model training completed in {duration}")

    # +---------------------+
    # |   Model inference   |
    # +---------------------+

    with open(
        f"{models_dir}/{args.model}_{args.particle}_models/tX_{label}.pkl", "rb"
    ) as file:
        x_scaler = pickle.load(file)

    x_post = invertColumnTransformer(x_scaler, x_val)

    with open(
        f"{models_dir}/{args.model}_{args.particle}_models/tY_{label}.pkl", "rb"
    ) as file:
        y_scaler = pickle.load(file)

    y_post = y_scaler.inverse_transform(y_val)

    output = model.generate(x_val, seed=None)
    out_post = y_scaler.inverse_transform(output)

    # +------------------+
    # |   Model export   |
    # +------------------+

    prefix = f"suid{study.study_id[:8]}-trial{trial.id:04d}"
    prefix += f"_{args.model}GAN-{args.particle}-{label}"

    export_model_fname = (
        f"{models_dir}/opt_studies/{args.model}_{args.particle}_models/{prefix}_model"
    )
    export_img_dirname = (
        f"{images_dir}/opt_studies/{args.model}_{args.particle}_img/{prefix}_img"
    )

    if args.saving:
        tf.saved_model.save(model.generator, export_dir=export_model_fname)
        hp.dump(
            f"{export_model_fname}/hyperparams.yml"
        )  # export also list of hyperparams
        np.savez(
            f"{export_model_fname}/results.npz",
            x=x_val,
            x_vars=x_vars,
            y=y_val,
            y_vars=y_vars,
            output=output,
        )  # export training results
        print(f"[INFO] Trained model correctly exported to {export_model_fname}")
        if not os.path.exists(export_img_dirname):
            os.makedirs(export_img_dirname)  # need to save images

    # +--------------------------------+
    # |   Feedbacks to Hopaas server   |
    # +--------------------------------+

    opt_scores = list()
    KS = KSDistance(dtype=DTYPE)

    for v in range(len(y_vars)):
        for i in range(len(P_BOUNDARIES) - 1):
            for j in range(len(ETA_BOUNDARIES) - 1):
                for k in range(len(NTRACKS_BOUNDARIES) - 1):
                    p_query = (x_post[:, 0] / 1e3 >= P_BOUNDARIES[i]) & (
                        x_post[:, 0] / 1e3 < P_BOUNDARIES[i + 1]
                    )
                    eta_query = (x_post[:, 1] >= ETA_BOUNDARIES[i]) & (
                        x_post[:, 1] < ETA_BOUNDARIES[i + 1]
                    )
                    ntracks_query = (x_post[:, 2] >= NTRACKS_BOUNDARIES[i]) & (
                        x_post[:, 2] < NTRACKS_BOUNDARIES[i + 1]
                    )

                    query = p_query & eta_query & ntracks_query
                    mean, std = np.mean(y_post[:, v][query]), np.std(
                        y_post[:, v][query]
                    )
                    min_ = mean - 4.0 * std
                    max_ = mean + 4.0 * std

                    score = KS(
                        x_true=y_post[:, v][query],
                        x_pred=out_post[:, v][query],
                        bins=np.linspace(min_, max_, 76),
                        weights_true=w_val[query] if w_val is not None else None,
                        weights_pred=w_val[query] if w_val is not None else None,
                        min_entries=250,
                    )

                    if score is not None:
                        opt_scores.append(score)

    final_opt_score = np.mean(opt_scores)
    trial.loss = final_opt_score

    print(
        f"[INFO] The trained model of Trial n. {trial.id} scored {final_opt_score:.3f}"
    )

    # +---------------------+
    # |   Training report   |
    # +---------------------+

    min_score = float(args.min_score_for_report)

    if final_opt_score <= min_score:
        report = Report()
        report.add_markdown(
            f'<h1 align="center">{args.model}GAN optimization report</h1>'
        )

        timestamp = str(datetime.now())
        date, hour = timestamp.split(" ")
        date = date.replace("-", "/")
        hour = hour.split(".")[0]

        info = [
            f"- Script executed on **{socket.gethostname()}** (address: {args.node_name})",
            f"- Trial **#{trial.id:04d}** (suid: {study.study_id})",
            f"- Model training completed in **{duration}**",
            f"- Report generated on **{date}** at **{hour}**",
            f"- Model trained on **{args.particle}** tracks",
        ]

        if args.fullsim:
            info += ["- Model trained on **detailed simulated** samples"]
        else:
            info += ["- Model trained on **calibration** samples"]
            if args.weights:
                info += ["- Any background components subtracted using **sWeights**"]
            else:
                info += ["- **sWeights not applied**"]

        report.add_markdown("\n".join([i for i in info]))

        report.add_markdown("---")

        ## Hyperparameters and other details
        report.add_markdown('<h2 align="center">Hyperparameters and other details</h2>')
        hyperparams = ""
        for k, v in hp.get_dict().items():
            hyperparams += f"- **{k}:** {v}\n"
        report.add_markdown(hyperparams)

        report.add_markdown("---")

        ## Generator architecture
        report.add_markdown('<h2 align="center">Generator architecture</h2>')
        report.add_markdown(f"**Model name:** {model.generator.name}")
        html_table, params_details = getSummaryHTML(model.generator)
        model_weights = ""
        for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
            model_weights += f"- **{k} params:** {n}\n"
        report.add_markdown(html_table)
        report.add_markdown(model_weights)

        report.add_markdown("---")

        ## Discriminator architecture
        report.add_markdown('<h2 align="center">Discriminator architecture</h2>')
        report.add_markdown(f"**Model name:** {model.discriminator.name}")
        html_table, params_details = getSummaryHTML(model.discriminator)
        model_weights = ""
        for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
            model_weights += f"- **{k} params:** {n}\n"
        report.add_markdown(html_table)
        report.add_markdown(model_weights)

        report.add_markdown("---")

        ## Training plots
        prepare_training_plots(
            report=report,
            history=train.history,
            metrics=metrics,
            num_epochs=EPOCHS,
            loss_name=model.loss_name,
            is_from_validation_set=(train_ratio != 1.0),
            save_images=args.saving,
            images_dirname=export_img_dirname,
        )

        ## Validation plots
        prepare_validation_plots(
            report=report,
            x_true=x_post,
            y_true=y_post,
            y_pred=out_post,
            y_vars=y_vars,
            weights=w_val,
            is_from_fullsim=args.fullsim,
            save_images=args.saving,
            images_dirname=export_img_dirname,
        )

        report_fname = f"{reports_dir}/opt_studies/{prefix}_train-report.html"
        report.write_report(filename=report_fname)
        print(f"[INFO] Training report correctly exported to {report_fname}")
