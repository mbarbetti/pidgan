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

from pidgan.callbacks.schedulers import LearnRateExpDecay
from pidgan.players.classifiers import ResClassifier
from pidgan.utils.preprocessing import invertColumnTransformer
from pidgan.utils.reports import initHPSingleton

DTYPE = np.float32
BATCHSIZE = 2048

here = os.path.abspath(os.path.dirname(__file__))

# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_training(model="isMuon", description="isMuonANN training setup")
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

npzfile = np.load(f"{data_dir}/isMuon-{args.particle}-{args.data_sample}-trainset.npz")

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

classifier = ResClassifier(
    num_hidden_layers=hp.get("num_hidden_layers", 10),
    mlp_hidden_units=hp.get("mlp_hidden_units", 128),
    mlp_hidden_activation=hp.get("mlp_hidden_activation", "relu"),
    mlp_hidden_kernel_regularizer=hp.get(
        "mlp_hidden_kernel_regularizer", k.regularizers.L2(5e-5)
    ),
    mlp_dropout_rates=hp.get("mlp_dropout_rates", 0.0),
    name="classifier",
    dtype=DTYPE,
)

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

opt = k.optimizers.Adam(hp.get("lr0", 0.001))
hp.get("optimizer", opt.name)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

hp.get("loss", "binary cross-entropy")
loss = k.losses.BinaryCrossentropy(label_smoothing=hp.get("label_smoothing", 0.05))

hp.get("metrics", ["auc"])
metrics = [k.metrics.AUC(name="auc")]

classifier.compile(
    optimizer=opt,
    loss=loss,
    metrics=metrics,
)

out = classifier(x[:BATCHSIZE])
classifier.summary()

# +--------------------------+
# |   Callbacks definition   |
# +--------------------------+

callbacks = list()

lr_sched = LearnRateExpDecay(
    classifier.optimizer,
    decay_rate=hp.get("decay_rate", 0.10),
    decay_steps=hp.get("decay_steps", 50_000),
    min_learning_rate=hp.get("min_learning_rate", 1e-6),
    verbose=True,
    key="lr",
)
hp.get("lr_sched", lr_sched.name)
callbacks.append(lr_sched)

# +------------------------+
# |   Training procedure   |
# +------------------------+

start = datetime.now()
train = classifier.fit(
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
    f"{models_dir}/isMuon_{args.particle}_models/tX_{args.data_sample}.pkl", "rb"
) as file:
    x_scaler = pickle.load(file)

x_post = invertColumnTransformer(x_scaler, x_val)

probs = classifier(x_val).numpy()
preds = probs > 0.5

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
model_name += f"_isMuon_{args.particle}_{args.data_sample}_ann"

export_model_dirname = f"{models_dir}/isMuon_{args.particle}_models/{model_name}"
export_img_dirname = f"{images_dir}/isMuon_{args.particle}_images/{model_name}"

if save_output:
    if not os.path.exists(export_model_dirname):
        os.makedirs(export_model_dirname)
    if not os.path.exists(export_img_dirname):
        os.makedirs(export_img_dirname)  # need to save images
    k.models.save_model(
        classifier.plain_keras,
        filepath=f"{export_model_dirname}/saved_model",
        save_format="tf",
    )
    print(f"[INFO] Trained model correctly exported to '{export_model_dirname}'")
    hp.dump(
        f"{export_model_dirname}/hyperparams.yml"
    )  # export also list of hyperparams
    pickle.dump(x_scaler, open(f"{export_model_dirname}/tX.pkl", "wb"))
    np.savez(
        f"{export_model_dirname}/results.npz",
        x_true=x_post,
        x_vars=x_vars,
        y_true=y_val,
        y_prob=probs,
        y_pred=preds,
        y_vars=y_vars,
    )  # export training results

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()

## Basic report info
fill_html_report(
    report=report,
    title="isMuonANN training report",
    train_duration=duration,
    report_datetime=(date, hour),
    particle=args.particle,
    data_sample=args.data_sample,
    trained_with_weights=args.weights,
    hp_dict=hp.get_dict(),
    model_arch=[classifier],
    model_labels=["Classifier"],
)

## Training plots
prepare_training_plots(
    report=report,
    model="isMuon",
    history=train.history,
    metrics=None,
    num_epochs=num_epochs,
    loss_name=None,
    from_validation_set=(train_ratio != 1.0),
    save_images=save_output,
    images_dirname=export_img_dirname,
)

## Validation plots
prepare_validation_plots(
    report=report,
    model="isMuon",
    x_true=x_post,
    y_true=y_val.flatten(),
    y_pred=probs.flatten(),
    y_vars=y_vars,
    weights=w_val,
    from_fullsim="calib" not in args.data_sample,
    save_images=save_output,
    images_dirname=export_img_dirname,
)

report_fname = f"{reports_dir}/{model_name}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to '{report_fname}'")
