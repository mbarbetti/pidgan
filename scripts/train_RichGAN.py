import os
import pickle
import socket
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from html_reports import Report
from sklearn.utils import shuffle
from utils_argparser import argparser_training
from utils_plot import (
    binned_validation_histogram,
    correlation_histogram,
    learn_rate_scheduling,
    learning_curves,
    metric_curves,
    validation_histogram,
)

from pidgan.algorithms import GAN
from pidgan.callbacks.schedulers import LearnRateExpDecay
from pidgan.players.discriminators import Discriminator
from pidgan.players.generators import Generator
from pidgan.utils.preprocessing import invertColumnTransformer
from pidgan.utils.reports import getSummaryHTML, initHPSingleton

PARTICLES = ["muon", "pion", "kaon", "proton"]
DTYPE = np.float32
BATCHSIZE = 256
EPOCHS = 100

# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_training(description="RichGAN training setup")
args = parser.parse_args()

# +-------------------+
# |   Initial setup   |
# +-------------------+

hp = initHPSingleton()

with open("config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
models_dir = config_dir["models_dir"]
images_dir = config_dir["images_dir"]
reports_dir = config_dir["reports_dir"]

chunk_size = int(args.chunk_size)
train_ratio = float(args.train_ratio)

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

discriminator = Discriminator(
    output_dim=hp.get("d_output_dim", 1),
    num_hidden_layers=hp.get("d_num_hidden_layers", 5),
    mlp_hidden_units=hp.get("d_mlp_hidden_units", 128),
    dropout_rate=hp.get("d_dropout_rate", 0.0),
    output_activation=hp.get("d_output_activation", "sigmoid"),
    name="discriminator",
    dtype=DTYPE,
)

model = GAN(
    generator=generator,
    discriminator=discriminator,
    use_original_loss=hp.get("gan_use_original_loss", True),
    injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.1),
)

output = model(x[:BATCHSIZE], y[:BATCHSIZE])
model.summary()

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

g_opt = tf.keras.optimizers.RMSprop(hp.get("g_lr0", 1e-4))
hp.get("g_optimizer", g_opt.name)

d_opt = tf.keras.optimizers.RMSprop(hp.get("d_lr0", 1e-4))
hp.get("d_optimizer", d_opt.name)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

model.compile(
    metrics=hp.get("metrics", ["accuracy", "bce"]),
    generator_optimizer=g_opt,
    discriminator_optimizer=d_opt,
    generator_upds_per_batch=hp.get("generator_upds_per_batch", 1),
    discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1),
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

output = model.generate(x_val, seed=None)

# +------------------+
# |   Model export   |
# +------------------+

timestamp = str(datetime.now())
date, hour = timestamp.split(" ")
date = date.replace("-", "/")
hour = hour.split(".")[0]

if args.test:
    prefix = "test"
else:
    prefix = ""
    timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
    for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
        prefix += time + unit  # YYYYMMDD-HHhMMmSSs
prefix += f"_RichGAN-{args.particle}"

export_model_fname = f"{models_dir}/Rich_{args.particle}_models/{prefix}_model"
export_img_dirname = f"{export_model_fname}/images"

if args.saving:
    tf.saved_model.save(model.generator, export_dir=export_model_fname)
    hp.dump(f"{export_model_fname}/hyperparams.yml")  # export also list of hyperparams
    np.savez(
        f"{export_model_fname}/results.npz",
        x=x_val,
        x_vars=x_vars,
        y=y_val,
        y_vars=y_vars,
        output=output,
    )  # export training results
    print(f"[INFO] Trained model correctly exported to {export_model_fname}")
    if not os.path.exists:
        os.makedirs(export_img_dirname)  # need to save images

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()
report.add_markdown('<h1 align="center">RichGAN training report</h1>')

info = [
    f"- Script executed on **{socket.gethostname()}**",
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
report.add_markdown('<h2 align="center">Training plots</h2>')

start_epoch = int(EPOCHS / 20)

#### Learning curves
learning_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    keys=["g_loss", "d_loss"],
    colors=["#3288bd", "#fc8d59"],
    labels=["generator", "discriminator"],
    legend_loc=None,
    save_figure=args.saving,
    scale_curves=False,
    export_fname=f"{export_img_dirname}/learn-curves",
)

#### Learning rate scheduling
learn_rate_scheduling(
    report=report,
    history=train.history,
    start_epoch=0,
    keys=["g_lr", "d_lr"],
    colors=["#3288bd", "#fc8d59"],
    labels=["generator", "discriminator"],
    legend_loc="upper right",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/lr-sched",
)

#### Generator loss
metric_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    key="g_loss",
    ylabel=model.loss_name,
    title="Generator learning curves",
    validation_set=(train_ratio != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc=None,
    yscale="linear",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/gen-loss",
)

#### Discriminator loss
metric_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    key="d_loss",
    ylabel=model.loss_name,
    title="Discriminator learning curves",
    validation_set=(train_ratio != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc=None,
    yscale="linear",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/disc-loss",
)

#### Accuracy curves
metric_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    key="accuracy",
    ylabel="Accuracy",
    title="Metric curves",
    validation_set=(train_ratio != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc=None,
    yscale="linear",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/accuracy-curves",
)

#### BCE curves
metric_curves(
    report=report,
    history=train.history,
    start_epoch=start_epoch,
    key="bce",
    ylabel="Binary cross-entropy",
    title="Metric curves",
    validation_set=(train_ratio != 1.0),
    colors=["#d01c8b", "#4dac26"],
    labels=["training set", "validation set"],
    legend_loc=None,
    yscale="linear",
    save_figure=args.saving,
    export_fname=f"{export_img_dirname}/bce-curves",
)

report.add_markdown("---")

## Validation plots
with open(f"{models_dir}/Rich_{args.particle}_models/tX.pkl", "rb") as file:
    x_scaler = pickle.load(file)

x_post = invertColumnTransformer(x_scaler, x_val)

with open(f"{models_dir}/Rich_{args.particle}_models/tY.pkl", "rb") as file:
    y_scaler = pickle.load(file)

y_post = y_scaler.inverse_transform(y_val)
out_post = y_scaler.inverse_transform(output)

for i, y_var in enumerate(y_vars):
    report.add_markdown(f'<h2 align="center">Validation plots of {y_var}</h2>')

    for log_scale in [False, True]:
        validation_histogram(
            report=report,
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}-hist",
        )

    for log_scale in [False, True]:
        correlation_histogram(
            report=report,
            data_corr=x_post[:, 0] / 1e3,
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            range_corr=[0, 100],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel="Momentum [GeV/$c$]",
            ylabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}_vs_p-corr_hist",
        )

    for log_scale in [False, True]:
        binned_validation_histogram(
            report=report,
            data_corr=x_post[:, 0] / 1e3,
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            boundaries_corr=[0.1, 5, 10, 25, 100],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            symbol_corr="$p$",
            unit_corr="[GeV/$c$]",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}-hist",
        )

    for log_scale in [False, True]:
        correlation_histogram(
            report=report,
            data_corr=x_post[:, 1],
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            range_corr=[2, 5],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel="Pseudorapidity",
            ylabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}_vs_eta-corr_hist",
        )

    for log_scale in [False, True]:
        binned_validation_histogram(
            report=report,
            data_corr=x_post[:, 1],
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            boundaries_corr=[1.8, 2.7, 3.5, 4.2, 5.5],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            symbol_corr="$\eta$",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}-hist",
        )

    for log_scale in [False, True]:
        correlation_histogram(
            report=report,
            data_corr=x_post[:, 2],
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            range_corr=[0, 500],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel="$\mathtt{nTracks}$",
            ylabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}_vs_nTracks-corr_hist",
        )

    for log_scale in [False, True]:
        binned_validation_histogram(
            report=report,
            data_corr=x_post[:, 2],
            data_ref=y_post[:, i],
            data_gen=out_post[:, i],
            boundaries_corr=[0, 50, 150, 300, 500],
            weights_ref=w_val,
            weights_gen=w_val,
            xlabel=f"{y_var}",
            label_ref="Full simulation" if args.fullsim else "Calibration samples",
            label_gen="GAN-based model",
            symbol_corr="$\mathtt{nTracks}$",
            log_scale=log_scale,
            save_figure=args.saving,
            export_fname=f"{export_img_dirname}/{y_var}-hist",
        )

    report.add_markdown("---")

report_fname = f"{reports_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to {report_fname}")
