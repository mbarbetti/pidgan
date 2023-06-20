import os
import pickle
from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import shuffle
from utils_argparser import argparser_preprocessing

MODELS = ["Rich", "Muon", "GlobalPID", "GlobalMuonId", "isMuon"]
PARTICLES = ["muon", "pion", "kaon", "proton"]
LABELS = ["sim", "calib"]


# +------------------+
# |   Parser setup   |
# +------------------+

parser = argparser_preprocessing(description="dataset preparation setup")
args = parser.parse_args()

# +-------------------+
# |   Initial setup   |
# +-------------------+

if "*" in args.filename:
    data_fnames = np.array(glob(args.filename))
else:
    data_fnames = np.array([args.filename])

max_files = int(args.max_files)
chunk_size = int(args.chunk_size)

indices = np.random.permutation(len(data_fnames))
data_fnames = data_fnames[indices][:max_files]

with open("config/directories.yml") as file:
    config_dir = yaml.full_load(file)

export_data_dir = config_dir["data_dir"]
images_dir = config_dir["images_dir"]
models_dir = config_dir["models_dir"]

export_model_fname = f"{models_dir}/{args.model}_{args.particle}_models"
export_img_dirname = f"{images_dir}/{args.model}_{args.particle}_images"

for dirname in [export_model_fname, export_img_dirname]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# +------------------+
# |   Data loading   |
# +------------------+

with open("config/selections.yml") as file:
    selections = yaml.full_load(file)[f"{args.model}"]

start = time()
dataframes = list()

for fname in data_fnames:
    with uproot.open(fname) as file:
        dataframes.append(
            file[f"PidTupler/pid_{args.particle}"]
            .arrays(library="pd")
            .query(" and ".join([s for s in selections]))
        )

print(f"[INFO] Data correctly loaded in {time()-start:.2f} s")

df = pd.concat(dataframes, ignore_index=True).dropna()
df = shuffle(df).reset_index(drop=True)[:chunk_size]
print(f"[INFO] DataFrame of {len(df)} rows correctly created")

# +---------------------------+
# |   Dataframe preparation   |
# +---------------------------+

with open("config/variables.yml") as file:
    vars_dict = yaml.full_load(file)[f"{args.model}"]

vars = vars_dict["x"] + vars_dict["y"]
if args.weights:
    vars += vars_dict["w"]
df = df[vars]

cols = dict()
x_vars = list()
y_vars = list()
w_var = list()
for v in vars:
    if "probe" in v:
        cols.update({f"{v}": v.split("_")[-1]})
    else:
        cols.update({f"{v}": v.split("_")[0]})
    if v in vars_dict["x"]:
        x_vars.append(cols[v])
    elif v in vars_dict["y"]:
        y_vars.append(cols[v])
    else:
        w_var.append(cols[v])
if len(w_var) == 0:
    w_var = None
df.rename(columns=cols, inplace=True)

if args.tricks:
    if args.model == "Rich":
        df["RichDLLpk"] = df["RichDLLp"] - df["RichDLLk"]
        y_vars.append("RichDLLpk")
        df.drop("RichDLLp", axis=1, inplace=True)
        y_vars.remove("RichDLLp")

    elif args.model == "Muon":
        df["MuonLL"] = df["MuonMuLL"] - df["MuonBgLL"]
        y_vars.append("MuonLL")
        df.drop("MuonBgLL", axis=1, inplace=True)
        y_vars.remove("MuonBgLL")

    elif args.model == "GlobalPID":
        df["PIDpk"] = df["PIDp"] - df["PIDK"]
        y_vars.append("PIDpk")
        df.drop("PIDp", axis=1, inplace=True)
        y_vars.remove("PIDp")

if args.verbose:
    print(df.describe())

# +------------------------+
# |   Data preprocessing   |
# +------------------------+

start = time()
df_preprocessed = df.copy()

x_features = list()
x_flags = list()
for v in x_vars:
    if v not in ["trackcharge", "isMuon"]:
        x_features.append(v)
    else:
        x_flags.append(v)

x_scaler = ColumnTransformer(
    [
        (
            "features",
            QuantileTransformer(output_distribution="normal"),
            np.arange(len(x_features)),
        ),
        ("flags", "passthrough", len(x_features) + np.arange(len(x_flags))),
    ]
).fit(df[x_features + x_flags].values)
df_preprocessed[x_features + x_flags] = x_scaler.transform(
    df[x_features + x_flags].values
)

y_features = list()
for v in y_vars:
    if v not in ["trackcharge", "isMuon"]:
        y_features.append(v)

if len(y_features) > 0:
    y_scaler = QuantileTransformer(output_distribution="normal").fit(
        df[y_features].values
    )
    df_preprocessed[y_features] = y_scaler.transform(df[y_features].values)

print(f"[INFO] Data preprocessing completed in {time()-start:.2f} s")

if args.verbose:
    print(df_preprocessed.describe())

# +--------------------------------+
# |   Input variables histograms   |
# +--------------------------------+

for var in x_features:
    min_ = df[var].values.min()
    max_ = df[var].values.max()
    bins = np.linspace(min_, max_, 101)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.title(f"{args.model} model - {args.particle} tracks", fontsize=14)
    plt.xlabel(f"{var}", fontsize=12)
    plt.ylabel("Candidates", fontsize=12)
    plt.hist(
        df[var].values,
        bins=bins,
        density=False,
        weights=df[w_var].values if args.weights else None,
        color="#3288bd",
        label="data",
    )
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(f"{export_img_dirname}/{var}-hist.png")
    plt.close()

# +---------------------------------+
# |   Output variables histograms   |
# +---------------------------------+

for var in y_features:
    min_ = df[var].values.min()
    max_ = df[var].values.max()
    bins = np.linspace(min_, max_, 101)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.title(f"{args.model} model - {args.particle} tracks", fontsize=14)
    plt.xlabel(f"{var}", fontsize=12)
    plt.ylabel("Candidates", fontsize=12)
    plt.hist(
        df[var].values,
        bins=bins,
        density=False,
        weights=df[w_var].values if args.weights else None,
        color="#3288bd",
        label="data",
    )
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(f"{export_img_dirname}/{var}-hist.png")
    plt.close()

# +--------------------------+
# |   Training data export   |
# +--------------------------+

export_data_fname = (
    f"{export_data_dir}/pidgan-{args.model}-{args.particle}-{args.label}-dataset"
)
npz_fname = f"{export_data_fname}.npz"
np.savez(
    file=npz_fname,
    x=df_preprocessed[x_vars].values,
    x_vars=x_vars,
    y=df_preprocessed[y_vars],
    y_vars=y_vars,
    w=df_preprocessed[w_var] if args.weights else None,
    w_var=w_var,
)
print(
    f"[INFO] Training data of {len(df_preprocessed)} instances correctly saved to {npz_fname}"
)

# +---------------------------------+
# |   Preprocessing models export   |
# +---------------------------------+

pkl_fname = f"{export_model_fname}/tX.pkl"
pickle.dump(x_scaler, open(pkl_fname, "wb"))
print(f"[INFO] Input variables scaler correctly saved to {pkl_fname}")

pkl_fname = f"{export_model_fname}/tY.pkl"
pickle.dump(y_scaler, open(pkl_fname, "wb"))
print(f"[INFO] Output variables scaler correctly saved to {pkl_fname}")