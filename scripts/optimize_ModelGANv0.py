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
from tensorflow import keras
from utils.utils_argparser import argparser_optimization
from utils.utils_training import prepare_training_plots, prepare_validation_plots

import pidgan
from pidgan.algorithms import (
    GAN,
    LSGAN,
    WGAN,
    WGAN_ALP,
    WGAN_GP,
    BceGAN,
    BceGAN_ALP,
    BceGAN_GP,
    CramerGAN,
)
from pidgan.callbacks.schedulers import LearnRateExpDecay
from pidgan.optimization.scores import KSDistance
from pidgan.players.classifiers import Classifier
from pidgan.players.discriminators import AuxDiscriminator
from pidgan.players.generators import Generator
from pidgan.utils.preprocessing import invertColumnTransformer
from pidgan.utils.reports import getSummaryHTML, initHPSingleton

DTYPE = np.float32
BATCHSIZE = 512
CHUNK_SIZE = 500_000
EPOCHS = 250

P_BOUNDARIES = [0, 5, 10, 25, 100]
ETA_BOUNDARIES = [1.8, 2.7, 3.5, 4.2, 5.5]
NTRACKS_BOUNDARIES = [0, 50, 150, 300, 500]

here = os.path.abspath(os.path.dirname(__file__))

# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_optimization(description="Model GAN optimization setup")
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

with open(f"{here}/config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
models_dir = config_dir["models_dir"]
images_dir = config_dir["images_dir"]
reports_dir = config_dir["reports_dir"]

vrs = "v0.0" if not args.test else "test"
num_epochs = EPOCHS if not args.test else 10
chunk_size = CHUNK_SIZE if not args.test else 50_000
train_ratio = float(args.train_ratio)

# +-----------------------------+
# |    Client initialization    |
# +-----------------------------+

with open(f"{here}/config/hopaas.yml") as file:
    config_hopaas = yaml.full_load(file)

server = config_hopaas["server"]
token = config_hopaas["token"]

client = hpc.Client(server=server, token=token)

# +----------------------+
# |    Study creation    |
# +----------------------+

properties = {
    "algo": hpc.suggestions.Categorical(
        [
            "gan",
            "bce-gan",
            "bce-gan-gp",
            "bce-gan-alp",
            "lsgan",
            "wgan",
            "wgan-gp",
            "cramer-gan",
            "wgan-alp",
        ]
    ),
    "g_lr0": hpc.suggestions.Float(1e-4, 1e-3),
    "d_lr0": hpc.suggestions.Float(1e-4, 1e-3),
    "g_dec_step": hpc.suggestions.Int(10_000, 200_000, step=5_000),
    "d_dec_step": hpc.suggestions.Int(10_000, 200_000, step=5_000),
}

properties.update(
    {
        "data_sample": args.data_sample,
        "train_ratio": train_ratio,
        "batch_size": BATCHSIZE,
        "epochs": num_epochs,
    }
)

study_name = f"{args.model}GAN-{args.particle}::GanAlgo::{vrs}"

study = hpc.Study(
    name=study_name,
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

    npzfile = np.load(
        f"{data_dir}/pidgan-{args.model}-{args.particle}-{args.data_sample}-data.npz"
    )

    x = npzfile["x"].astype(DTYPE)[:chunk_size]
    x_vars = [str(v) for v in npzfile["x_vars"]]
    y = npzfile["y"].astype(DTYPE)[:chunk_size]
    y_vars = [str(v) for v in npzfile["y_vars"]]
    if "sim" not in args.data_sample:
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
        mlp_dropout_rates=hp.get("g_mlp_dropout_rates", 0.0),
        name="generator",
        dtype=DTYPE,
    )

    d_output_dim = 1 if trial.algo != "cramer-gan" else 256
    d_output_activation = (
        "sigmoid" if trial.algo in ["gan", "bce-gan", "lsgan"] else None
    )
    if args.model == "Rich":
        d_aux_features = [f"{y_vars.index('RichDLLp')} - {y_vars.index('RichDLLk')}"]
    elif args.model == "Muon":
        d_aux_features = [f"{y_vars.index('MuonMuLL')} - {y_vars.index('MuonBgLL')}"]
    else:
        d_aux_features = [f"{y_vars.index('PIDp')} - {y_vars.index('PIDK')}"]

    discriminator = AuxDiscriminator(
        output_dim=hp.get("d_output_dim", d_output_dim),
        aux_features=hp.get("d_aux_features", d_aux_features),
        num_hidden_layers=hp.get("d_num_hidden_layers", 5),
        mlp_hidden_units=hp.get("d_mlp_hidden_units", 128),
        mlp_dropout_rates=hp.get("d_mlp_dropout_rates", 0.1),
        output_activation=hp.get("d_output_activation", d_output_activation),
        name="discriminator",
        dtype=DTYPE,
    )

    referee = Classifier(
        num_hidden_layers=hp.get("r_num_hidden_layers", 5),
        mlp_hidden_units=hp.get("r_mlp_hidden_units", 128),
        mlp_dropout_rates=hp.get("r_mlp_dropout_rates", 0.1),
        name="referee",
        dtype=DTYPE,
    )

    if trial.algo == "gan":
        gan = GAN(
            generator=generator,
            discriminator=discriminator,
            use_original_loss=hp.get("gan_use_original_loss", True),
            injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.05),
            referee=referee,
        )
    elif trial.algo == "bce-gan":
        gan = BceGAN(
            generator=generator,
            discriminator=discriminator,
            from_logits=hp.get("gan_from_logits", False),
            label_smoothing=hp.get("gan_label_smoothing", 0.1),
            injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.05),
            referee=referee,
        )
    elif trial.algo == "bce-gan-gp":
        gan = BceGAN_GP(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", 1.0),
            lipschitz_penalty_strategy=hp.get(
                "gan_lipschitz_penalty_strategy", "two-sided"
            ),
            referee=referee,
        )
    elif trial.algo == "bce-gan-alp":
        gan = BceGAN_ALP(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", 1.0),
            lipschitz_penalty_strategy=hp.get(
                "gan_lipschitz_penalty_strategy", "one-sided"
            ),
            referee=referee,
        )
    elif trial.algo == "lsgan":
        gan = LSGAN(
            generator=generator,
            discriminator=discriminator,
            minimize_pearson_chi2=hp.get("gan_minimize_pearson_chi2", False),
            referee=referee,
        )
    elif trial.algo == "wgan":
        gan = WGAN(
            generator=generator,
            discriminator=discriminator,
            clip_param=hp.get("gan_clip_param", 0.01),
            referee=referee,
        )
    elif trial.algo == "wgan-gp":
        gan = WGAN_GP(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", 1.0),
            lipschitz_penalty_strategy=hp.get(
                "gan_lipschitz_penalty_strategy", "two-sided"
            ),
            referee=referee,
        )
    elif trial.algo == "cramer-gan":
        gan = CramerGAN(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", 1.0),
            lipschitz_penalty_strategy=hp.get(
                "gan_lipschitz_penalty_strategy", "two-sided"
            ),
            referee=referee,
        )
    elif trial.algo == "wgan-alp":
        gan = WGAN_ALP(
            generator=generator,
            discriminator=discriminator,
            lipschitz_penalty=hp.get("gan_lipschitz_penalty", 1.0),
            lipschitz_penalty_strategy=hp.get(
                "gan_lipschitz_penalty_strategy", "one-sided"
            ),
            referee=referee,
        )
    hp.get("gan_name", gan.name)

    output = gan(x[:BATCHSIZE], y[:BATCHSIZE])
    gan.summary()

    # +----------------------+
    # |   Optimizers setup   |
    # +----------------------+

    g_opt = keras.optimizers.RMSprop(hp.get("g_lr0", trial.g_lr0))
    hp.get("g_optimizer", "RMSprop")

    d_opt = keras.optimizers.RMSprop(hp.get("d_lr0", trial.d_lr0))
    hp.get("d_optimizer", "RMSprop")

    r_opt = keras.optimizers.RMSprop(hp.get("r_lr0", 0.001))
    hp.get("r_optimizer", "RMSprop")

    # +----------------------------+
    # |   Training configuration   |
    # +----------------------------+

    metrics = (
        ["accuracy", "bce"]
        if trial.algo in ["gan", "bce-gan", "lsgan"]
        else ["wass_dist"]
    )

    gan.compile(
        metrics=hp.get("metrics", metrics),
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        generator_upds_per_batch=hp.get("generator_upds_per_batch", 1),
        discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 2),
        referee_optimizer=r_opt,
        referee_upds_per_batch=hp.get("referee_upds_per_batch", 1),
    )

    # +--------------------------+
    # |   Callbacks definition   |
    # +--------------------------+

    callbacks = list()

    g_sched = LearnRateExpDecay(
        gan.generator_optimizer,
        decay_rate=hp.get("g_decay_rate", 0.10),
        decay_steps=hp.get("g_decay_steps", trial.g_dec_step),
        min_learning_rate=hp.get("g_min_learning_rate", trial.g_lr0 / 1e3),
        verbose=True,
        key="g_lr",
    )
    hp.get("g_sched", g_sched.name)
    callbacks.append(g_sched)

    d_sched = LearnRateExpDecay(
        gan.discriminator_optimizer,
        decay_rate=hp.get("d_decay_rate", 0.10),
        decay_steps=hp.get("d_decay_steps", trial.d_dec_step),
        min_learning_rate=hp.get("d_min_learning_rate", trial.d_lr0 / 1e3),
        verbose=True,
        key="d_lr",
    )
    hp.get("d_sched", d_sched.name)
    callbacks.append(d_sched)

    r_sched = LearnRateExpDecay(
        gan.referee_optimizer,
        decay_rate=hp.get("r_decay_rate", 0.10),
        decay_steps=hp.get("r_decay_steps", 100_000),
        min_learning_rate=hp.get("r_min_learning_rate", 1e-6),
        verbose=True,
        key="r_lr",
    )
    hp.get("r_sched", r_sched.name)
    callbacks.append(r_sched)

    # +------------------------+
    # |   Training procedure   |
    # +------------------------+

    start = datetime.now()
    train = gan.fit(
        train_ds,
        epochs=hp.get("epochs", num_epochs),
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
        f"{models_dir}/{args.model}_{args.particle}_models/tX_{args.data_sample}.pkl",
        "rb",
    ) as file:
        x_scaler = pickle.load(file)

    x_post = invertColumnTransformer(x_scaler, x_val)

    with open(
        f"{models_dir}/{args.model}_{args.particle}_models/tY_{args.data_sample}.pkl",
        "rb",
    ) as file:
        y_scaler = pickle.load(file)

    y_post = y_scaler.inverse_transform(y_val)

    out = gan.generate(x_val, seed=None)
    out_post = y_scaler.inverse_transform(out)

    # +--------------------------------+
    # |   Feedbacks to Hopaas server   |
    # +--------------------------------+

    KS = KSDistance(dtype=DTYPE)

    bin_scores = list()
    bin_entries = list()
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
                entries = np.count_nonzero(query)

                r_out_true = gan.referee((x_val[query], y_val[query]))
                r_out_pred = gan.referee((x_val[query], out[query]))

                score = KS(
                    x_true=r_out_true,
                    x_pred=r_out_pred,
                    bins=np.linspace(0.0, 1.0, 101),
                    weights_true=w_val[query] if w_val is not None else None,
                    weights_pred=w_val[query] if w_val is not None else None,
                    min_entries=500,
                )

                if score is not None:
                    bin_scores.append(score)
                    bin_entries.append(entries)

    bin_scores = np.array(bin_scores)
    bin_entries = np.array(bin_entries)
    opt_score = np.sum(bin_scores * bin_entries) / np.sum(bin_entries)
    trial.loss = opt_score

    print(f"[INFO] The trained model of Trial n. {trial.id} scored {opt_score:.3f}")

    # +------------------+
    # |   Model export   |
    # +------------------+

    min_score = float(args.min_score_for_report)

    prefix = f"suid{study.study_id[:8]}-trial{trial.id:04d}"
    prefix += f"_{args.model}GAN-{args.particle}-{args.data_sample}"

    export_model_dirname = (
        f"{models_dir}/{args.model}_{args.particle}_models/{prefix}_model"
    )
    export_img_dirname = f"{images_dir}/{args.model}_{args.particle}_img/{prefix}_img"

    if args.saving or opt_score <= min_score / 10.0:
        if not os.path.exists(export_model_dirname):
            os.makedirs(export_model_dirname)
        if not os.path.exists(export_img_dirname):
            os.makedirs(export_img_dirname)  # need to save images
        keras.models.save_model(
            gan.generator.export_model,
            filepath=f"{export_model_dirname}/saved_generator",
            save_format="tf",
        )
        print(f"[INFO] Trained model correctly exported to '{export_model_dirname}'")
        hp.dump(
            f"{export_model_dirname}/hyperparams.yml"
        )  # export also list of hyperparams
        pickle.dump(x_scaler, open(f"{export_model_dirname}/tX.pkl", "wb"))
        pickle.dump(y_scaler, open(f"{export_model_dirname}/tY.pkl", "wb"))
        np.savez(
            f"{export_model_dirname}/results.npz",
            x=x_val,
            x_vars=x_vars,
            y=y_val,
            y_vars=y_vars,
            output=output,
        )  # export training results

    # +---------------------+
    # |   Training report   |
    # +---------------------+

    if opt_score <= min_score:
        report = Report()
        report.add_markdown(
            f'<h1 align="center">{args.model}GAN optimization report</h1>'
        )

        timestamp = str(datetime.now())
        date, hour = timestamp.split(" ")
        date = date.replace("-", "/")
        hour = hour.split(".")[0]

        info = [
            f"- Script executed on **{socket.gethostname()}** "
            f"(address: {args.node_name})",
            f"- Trial **#{trial.id:04d}** (suid: {study.study_id})",
            f"- Optimization score (K-S distance): **{opt_score:.3f}**",
            f"- Model training completed in **{duration}**",
            f"- Model training executed with **TF{tf.__version__}** "
            f"and **pidgan v{pidgan.__version__}**",
            f"- Report generated on **{date}** at **{hour}**",
            f"- Model trained on **{args.particle}** tracks",
        ]

        if "sim" in args.data_sample:
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
        report.add_markdown(f"**Model name:** {gan.generator.name}")
        html_table, params_details = getSummaryHTML(gan.generator.export_model)
        model_weights = ""
        for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
            model_weights += f"- **{k} params:** {n}\n"
        report.add_markdown(html_table)
        report.add_markdown(model_weights)

        report.add_markdown("---")

        ## Discriminator architecture
        report.add_markdown('<h2 align="center">Discriminator architecture</h2>')
        report.add_markdown(f"**Model name:** {gan.discriminator.name}")
        html_table, params_details = getSummaryHTML(gan.discriminator.export_model)
        model_weights = ""
        for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
            model_weights += f"- **{k} params:** {n}\n"
        report.add_markdown(html_table)
        report.add_markdown(model_weights)

        report.add_markdown("---")

        ## Referee architecture
        report.add_markdown('<h2 align="center">Referee architecture</h2>')
        report.add_markdown(f"**Model name:** {gan.referee.name}")
        html_table, params_details = getSummaryHTML(gan.referee.export_model)
        model_weights = ""
        for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
            model_weights += f"- **{k} params:** {n}\n"
        report.add_markdown(html_table)
        report.add_markdown(model_weights)

        report.add_markdown("---")

        ## Training plots
        prepare_training_plots(
            report=report,
            model=args.model,
            history=train.history,
            metrics=metrics,
            num_epochs=num_epochs,
            loss_name=gan.loss_name,
            from_validation_set=(train_ratio != 1.0),
            referee_available=True,
            save_images=args.saving,
            images_dirname=export_img_dirname,
        )

        ## Validation plots
        prepare_validation_plots(
            report=report,
            model=args.model,
            x_true=x_post,
            y_true=y_post,
            y_pred=out_post,
            y_vars=y_vars,
            weights=w_val,
            from_fullsim="sim" in args.data_sample,
            save_images=args.saving,
            images_dirname=export_img_dirname,
        )

        report_fname = f"{reports_dir}/opt_studies/{prefix}_train-report.html"
        report.write_report(filename=report_fname)
        print(f"[INFO] Training report correctly exported to '{report_fname}'")
