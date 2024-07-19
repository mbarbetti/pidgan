import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import yaml
import pickle
import keras as k
import numpy as np
import tensorflow as tf

from datetime import datetime
from html_reports import Report
from sklearn.utils import shuffle
from utils.utils_argparser import argparser_training
from utils.utils_training import (
    fill_html_report,
    prepare_training_plots,
    prepare_validation_plots,
)

from pidgan.algorithms import BceGAN
from pidgan.callbacks.schedulers import LearnRateExpDecay
from pidgan.players.classifiers import Classifier
from pidgan.players.discriminators import AuxDiscriminator
from pidgan.players.generators import ResGenerator
from pidgan.utils.preprocessing import invertColumnTransformer
from pidgan.utils.reports import initHPSingleton

DTYPE = np.float32
BATCHSIZE = 2048

here = os.path.abspath(os.path.dirname(__file__))

# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_training(model="Rich", description="RichGAN training setup")
args = parser.parse_args()

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

num_epochs = int(args.num_epochs)
chunk_size = int(args.chunk_size)
train_ratio = float(args.train_ratio)

# +------------------+
# |   Data loading   |
# +------------------+

npzfile = np.load(f"{data_dir}/Rich-{args.particle}-{args.data_sample}-trainset.npz")

x = npzfile["x"].astype(DTYPE)[:chunk_size]
x_vars = [str(v) for v in npzfile["x_vars"]]
y = npzfile["y"].astype(DTYPE)[:chunk_size]
y_vars = [str(v) for v in npzfile["y_vars"]]
if "calib" in args.data_sample:
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

chunk_size = hp.get("chunk_size", x.shape[0])
train_size = hp.get("train_size", int(train_ratio * chunk_size))

# +-------------------------+
# |   Dataset preparation   |
# +-------------------------+

x_train = x[:train_size]
y_train = y[:train_size]
if w is not None:
    w_train = w[:train_size]
    train_slices = (x_train, y_train, w_train)
else:
    w_train = None
    train_slices = (x_train, y_train)
train_ds = (
    tf.data.Dataset.from_tensor_slices(train_slices)
    .batch(hp.get("batch_size", BATCHSIZE), drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

if train_ratio != 1.0:
    x_val = x[train_size:]
    y_val = y[train_size:]
    if w is not None:
        w_val = w[train_size:]
        val_slices = (x_val, y_val, w_val)
    else:
        w_val = None
        val_slices = (x_val, y_val)
    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_slices)
        .batch(BATCHSIZE, drop_remainder=True)
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

generator = ResGenerator(
    output_dim=hp.get("g_output_dim", y.shape[1]),
    latent_dim=hp.get("g_latent_dim", 64),
    num_hidden_layers=hp.get("g_num_hidden_layers", 10),
    mlp_hidden_units=hp.get("g_mlp_hidden_units", 128),
    mlp_dropout_rates=hp.get("g_mlp_dropout_rates", 0.0),
    output_activation=hp.get("g_output_activation", None),
    name="generator",
    dtype=DTYPE,
)

discriminator = AuxDiscriminator(
    output_dim=hp.get("d_output_dim", 1),
    aux_features=hp.get(
        "d_aux_features",
        [
            f"{y_vars.index('RichDLLmu')} - {y_vars.index('RichDLLe')}"
            f"{y_vars.index('RichDLLp')} - {y_vars.index('RichDLLk')}"
        ],
    ),
    num_hidden_layers=hp.get("d_num_hidden_layers", 10),
    mlp_hidden_units=hp.get("d_mlp_hidden_units", 128),
    mlp_dropout_rates=hp.get("d_mlp_dropout_rates", 0.0),
    enable_residual_blocks=hp.get("d_enable_residual_blocks", True),
    output_activation=hp.get("d_output_activation", "sigmoid"),
    name="discriminator",
    dtype=DTYPE,
)

referee = Classifier(
    num_hidden_layers=hp.get("r_num_hidden_layers", 5),
    mlp_hidden_units=hp.get("r_mlp_hidden_units", 128),
    mlp_hidden_activation=hp.get("r_mlp_hidden_activation", "relu"),
    mlp_hidden_kernel_regularizer=hp.get(
        "r_mlp_hidden_kernel_regularizer", k.regularizers.L2(5e-5)
    ),
    mlp_dropout_rates=hp.get("r_mlp_dropout_rates", 0.0),
    name="referee",
    dtype=DTYPE,
)

gan = BceGAN(
    generator=generator,
    discriminator=discriminator,
    from_logits=hp.get("gan_from_logits", False),
    label_smoothing=hp.get("gan_label_smoothing", 0.01),
    injected_noise_stddev=hp.get("gan_injected_noise_stddev", 0.02),
    feature_matching_penalty=hp.get("gan_feature_matching_penalty", 0.0),
    referee=referee if args.referee else None,
)
hp.get("gan_name", gan.name)

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

g_opt = k.optimizers.RMSprop(hp.get("g_lr0", 4e-4))
hp.get("g_optimizer", g_opt.name)

d_opt = k.optimizers.RMSprop(hp.get("d_lr0", 5e-4))
hp.get("d_optimizer", d_opt.name)

if gan.referee is not None:
    r_opt = k.optimizers.RMSprop(hp.get("r_lr0", 1e-3))
    hp.get("r_optimizer", r_opt.name)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

metrics = ["accuracy", "bce"]

gan.compile(
    metrics=hp.get("metrics", metrics),
    generator_optimizer=g_opt,
    discriminator_optimizer=d_opt,
    generator_upds_per_batch=hp.get("generator_upds_per_batch", 1),
    discriminator_upds_per_batch=hp.get("discriminator_upds_per_batch", 1),
    referee_optimizer=r_opt if gan.referee else None,
    referee_upds_per_batch=hp.get("referee_upds_per_batch", 1) if gan.referee else None,
)

out = gan(x[:BATCHSIZE], y[:BATCHSIZE])
gan.summary()

# +--------------------------+
# |   Callbacks definition   |
# +--------------------------+

callbacks = list()

g_lr_sched = LearnRateExpDecay(
    gan.generator_optimizer,
    decay_rate=hp.get("g_decay_rate", 0.10),
    decay_steps=hp.get("g_decay_steps", 125_000),
    min_learning_rate=hp.get("g_min_learning_rate", 1e-6),
    verbose=True,
    key="g_lr",
)
hp.get("g_sched", g_lr_sched.name)
callbacks.append(g_lr_sched)

d_lr_sched = LearnRateExpDecay(
    gan.discriminator_optimizer,
    decay_rate=hp.get("d_decay_rate", 0.10),
    decay_steps=hp.get("d_decay_steps", 200_000),
    min_learning_rate=hp.get("d_min_learning_rate", 1e-6),
    verbose=True,
    key="d_lr",
)
hp.get("d_sched", d_lr_sched.name)
callbacks.append(d_lr_sched)

if gan.referee is not None:
    r_lr_sched = LearnRateExpDecay(
        gan.referee_optimizer,
        decay_rate=hp.get("r_decay_rate", 0.10),
        decay_steps=hp.get("r_decay_steps", 150_000),
        min_learning_rate=hp.get("r_min_learning_rate", 1e-6),
        verbose=True,
        key="r_lr",
    )
    hp.get("r_sched", r_lr_sched.name)
    callbacks.append(r_lr_sched)

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = gan.fit(
    train_ds,
    epochs=hp.get("num_epochs", num_epochs),
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
    f"{models_dir}/Rich_{args.particle}_models/tX_{args.data_sample}.pkl", "rb"
) as file:
    x_scaler = pickle.load(file)

x_post = invertColumnTransformer(x_scaler, x_val)

with open(
    f"{models_dir}/Rich_{args.particle}_models/tY_{args.data_sample}.pkl", "rb"
) as file:
    y_scaler = pickle.load(file)

y_post = y_scaler.inverse_transform(y_val)

out = gan.generate(x_val, seed=None)
out_post = y_scaler.inverse_transform(out)

# +------------------+
# |   Model export   |
# +------------------+

timestamp = str(datetime.now())
date, hour = timestamp.split(" ")
date = date.replace("-", "/")
hour = hour.split(".")[0]

if args.test:
    model_name = "test"
    save_output = args.saving
elif args.latest:
    model_name = "latest"
    save_output = True
else:
    model_name = ""
    save_output = args.saving
    timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
    for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
        model_name += time + unit  # YYYYMMDD-HHhMMmSSs
model_name += f"_Rich_{args.particle}_{args.data_sample}_gan"

export_model_dirname = f"{models_dir}/Rich_{args.particle}_models/{model_name}"
export_img_dirname = f"{images_dir}/Rich_{args.particle}_images/{model_name}"

if save_output:
    if not os.path.exists(export_model_dirname):
        os.makedirs(export_model_dirname)
    if not os.path.exists(export_img_dirname):
        os.makedirs(export_img_dirname)  # need to save images
    k.models.save_model(
        gan.generator.plain_keras,
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
        x_true=x_post,
        x_vars=x_vars,
        y_true=y_post,
        y_pred=out_post,
        y_vars=y_vars,
    )  # export training results

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()

## Basic report info
fill_html_report(
    report=report,
    title="RichGAN training report",
    train_duration=duration,
    report_datetime=(date, hour),
    particle=args.particle,
    data_sample=args.data_sample,
    trained_with_weights=args.weights,
    hp_dict=hp.get_dict(),
    model_arch=[gan.generator, gan.discriminator, gan.referee],
    model_labels=["Generator", "Discriminator", "Referee"],
)

## Training plots
prepare_training_plots(
    report=report,
    model="Rich",
    history=train.history,
    metrics=metrics,
    num_epochs=num_epochs,
    loss_name=gan.loss_name,
    from_validation_set=(train_ratio != 1.0),
    referee_available=gan.referee is not None,
    save_images=save_output,
    images_dirname=export_img_dirname,
)

## Validation plots
prepare_validation_plots(
    report=report,
    model="Rich",
    x_true=x_post,
    y_true=y_post,
    y_pred=out_post,
    y_vars=y_vars,
    weights=w_val,
    from_fullsim="calib" not in args.data_sample,
    save_images=save_output,
    images_dirname=export_img_dirname,
)

report_fname = f"{reports_dir}/{model_name}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to '{report_fname}'")
