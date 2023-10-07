import os
import pickle
import socket
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
from html_reports import Report
from sklearn.utils import shuffle
from tensorflow import keras
from utils.utils_argparser import argparser_training
from utils.utils_training import prepare_training_plots, prepare_validation_plots

import pidgan
from pidgan.callbacks.schedulers import LearnRateExpDecay
from pidgan.players.classifiers import Classifier
from pidgan.utils.preprocessing import invertColumnTransformer
from pidgan.utils.reports import getSummaryHTML, initHPSingleton

DTYPE = np.float32
BATCHSIZE = 512

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

npzfile = np.load(
    f"{data_dir}/pidgan-isMuon-{args.particle}-{args.data_sample}-data.npz"
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
    .cache()
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

classifier = Classifier(
    num_hidden_layers=hp.get("num_hidden_layers", 5),
    mlp_hidden_units=hp.get("mlp_hidden_units", 128),
    dropout_rate=hp.get("dropout_rate", 0.1),
    name="classifier",
    dtype=DTYPE,
)

out = classifier(x[:BATCHSIZE])
classifier.summary()

# +----------------------+
# |   Optimizers setup   |
# +----------------------+

opt = keras.optimizers.Adam(hp.get("lr0", 0.001))
hp.get("optimizer", opt.name)

# +----------------------------+
# |   Training configuration   |
# +----------------------------+

classifier.compile(optimizer=opt, loss=hp.get("loss", "bce"), metrics=None)

# +--------------------------+
# |   Callbacks definition   |
# +--------------------------+

callbacks = list()

lr_sched = LearnRateExpDecay(
    classifier.optimizer,
    decay_rate=hp.get("decay_rate", 0.10),
    decay_steps=hp.get("decay_steps", 100_000),
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
    prefix = "test"
    save_output = args.saving
elif args.latest:
    prefix = "latest"
    save_output = True
else:
    prefix = ""
    save_output = args.saving
    timestamp = timestamp.split(".")[0].replace("-", "").replace(" ", "-")
    for time, unit in zip(timestamp.split(":"), ["h", "m", "s"]):
        prefix += time + unit  # YYYYMMDD-HHhMMmSSs
prefix += f"_isMuonANN-{args.particle}_{args.data_sample}"

export_model_dirname = f"{models_dir}/isMuon_{args.particle}_models/{prefix}_model"
export_img_dirname = f"{images_dir}/isMuon_{args.particle}_img/{prefix}_img"

if save_output:
    if not os.path.exists(export_model_dirname):
        os.makedirs(export_model_dirname)
    if not os.path.exists(export_img_dirname):
        os.makedirs(export_img_dirname)  # need to save images
    keras.models.save_model(
        classifier.export_model,
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
        x=x_val,
        x_vars=x_vars,
        y=y_val,
        y_vars=y_vars,
        probs=probs,
        preds=preds,
    )  # export training results

# +---------------------+
# |   Training report   |
# +---------------------+

report = Report()
report.add_markdown('<h1 align="center">isMuonANN training report</h1>')

info = [
    f"- Script executed on **{socket.gethostname()}**",
    f"- Model training completed in **{duration}**",
    f"- Model training executed with **pidgan v{pidgan.__version__}**",
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

## Classifier architecture
report.add_markdown('<h2 align="center">Classifier architecture</h2>')
report.add_markdown(f"**Model name:** {classifier.name}")
html_table, params_details = getSummaryHTML(classifier.export_model)
model_weights = ""
for k, n in zip(["Total", "Trainable", "Non-trainable"], params_details):
    model_weights += f"- **{k} params:** {n}\n"
report.add_markdown(html_table)
report.add_markdown(model_weights)

report.add_markdown("---")

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
    from_fullsim="sim" in args.data_sample,
    save_images=save_output,
    images_dirname=export_img_dirname,
)

report_fname = f"{reports_dir}/{prefix}_train-report.html"
report.write_report(filename=report_fname)
print(f"[INFO] Training report correctly exported to '{report_fname}'")
