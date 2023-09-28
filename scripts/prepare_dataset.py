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
from tqdm import trange
from utils.utils_argparser import argparser_preprocessing

here = os.path.abspath(os.path.dirname(__file__))

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

with open(f"{here}/config/directories.yml") as file:
    config_dir = yaml.full_load(file)

export_data_dir = config_dir["data_dir"]
images_dir = config_dir["images_dir"]
models_dir = config_dir["models_dir"]

export_model_fname = f"{models_dir}/{args.model}_{args.particle}_models"
export_img_dirname = f"{images_dir}/{args.model}_{args.particle}_img"

for dirname in [export_model_fname, export_img_dirname]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# +------------------+
# |   Data loading   |
# +------------------+

with open(f"{here}/config/selections.yml") as file:
    selections = yaml.full_load(file)[f"{args.model}"]

start = time()
dataframes = list()

for i in trange(len(data_fnames), desc="Processing data files", unit="file"):
    with uproot.open(data_fnames[i]) as file:
        dataframes.append(
            file[f"PidTupler/pid_{args.particle}"]
            .arrays(library="pd")
            .query(" and ".join([s for s in selections]))
        )

print(
    f"[INFO] Data from {len(data_fnames)} files correctly loaded in {time()-start:.2f} s"
)

df = pd.concat(dataframes, ignore_index=True).dropna()
df = shuffle(df).reset_index(drop=True)[:chunk_size]
print(f"[INFO] DataFrame of {len(df)} rows correctly created")

# +---------------------------+
# |   Dataframe preparation   |
# +---------------------------+

with open(f"{here}/config/variables.yml") as file:
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

    elif "GlobalPID" in args.model:
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
col_features = list()
x_flags = list()
col_flags = list()
for i, v in enumerate(x_vars):
    if v not in ["trackcharge"]:
        x_features.append(v)
        col_features.append(i)
    else:
        x_flags.append(v)
        col_flags.append(i)
new_x_vars = x_features + x_flags

x_scaler = ColumnTransformer(
    [
        (
            "features",
            QuantileTransformer(n_quantiles=1000, output_distribution="normal"),
            col_features,
        ),
        ("flags", "passthrough", col_flags),
    ]
).fit(df[x_vars].values)
df_preprocessed[new_x_vars] = x_scaler.transform(df[x_vars].values)

y_features = list()
for v in y_vars:
    if v not in ["isMuon"]:
        y_features.append(v)

if len(y_features) > 0:
    y_scaler = QuantileTransformer(n_quantiles=1000, output_distribution="normal").fit(
        df[y_features].values
    )
    df_preprocessed[y_features] = y_scaler.transform(df[y_features].values)
    y_vars = y_features
else:
    y_scaler = None
    y_vars = ["isMuon"]

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

    plt.figure(figsize=(16, 5))

    for i, yscale in enumerate(["linear", "log"]):
        plt.subplot(1, 2, i + 1)
        plt.xlabel(f"{var}", fontsize=12)
        plt.ylabel("Candidates", fontsize=12)
        plt.hist(
            df[var].values,
            bins=bins,
            weights=df[w_var].values if args.weights else None,
            color="#3288bd",
            label="data",
        )
        plt.yscale(yscale)
        plt.legend(title=f"{args.particle} tracks".capitalize(), fontsize=10)

    fig_fname = f"{export_img_dirname}/{var}-hist-{args.data_sample}.png"
    plt.savefig(fig_fname, dpi=300)

    if args.verbose:
        print(f"[INFO] Figure correctly exported to '{fig_fname}'")

    plt.close()

# +---------------------------------+
# |   Output variables histograms   |
# +---------------------------------+

if args.model != "isMuon":
    for var in y_features:
        min_ = df[var].values.min()
        max_ = df[var].values.max()
        bins = np.linspace(min_, max_, 101)

        plt.figure(figsize=(16, 5))

        for i, yscale in enumerate(["linear", "log"]):
            plt.subplot(1, 2, i + 1)
            plt.xlabel(f"{var}", fontsize=12)
            plt.ylabel("Candidates", fontsize=12)
            plt.hist(
                df[var].values,
                bins=bins,
                weights=df[w_var].values if args.weights else None,
                color="#3288bd",
                label="data",
            )
            plt.yscale(yscale)
            plt.legend(title=f"{args.particle} tracks".capitalize(), fontsize=10)

        fig_fname = f"{export_img_dirname}/{var}-hist-{args.data_sample}.png"
        plt.savefig(fig_fname, dpi=300)

        if args.verbose:
            print(f"[INFO] Figure correctly exported to '{fig_fname}'")

        plt.close()

# +--------------------------+
# |   Training data export   |
# +--------------------------+

export_data_fname = (
    f"{export_data_dir}/pidgan-{args.model}-{args.particle}-{args.data_sample}-data.npz"
)

np.savez(
    file=export_data_fname,
    x=df_preprocessed[new_x_vars].values,
    x_vars=new_x_vars,
    y=df_preprocessed[y_vars].values,
    y_vars=y_vars,
    w=df_preprocessed[w_var].values if args.weights else None,
    w_var=w_var,
)

print(
    f"[INFO] Training data of {len(df_preprocessed)} instances correctly saved to '{export_data_fname}'"
)

# +---------------------------------+
# |   Preprocessing models export   |
# +---------------------------------+

pkl_fname = f"{export_model_fname}/tX_{args.data_sample}.pkl"
pickle.dump(x_scaler, open(pkl_fname, "wb"))
print(f"[INFO] Input variables scaler correctly saved to '{pkl_fname}'")

if y_scaler is not None:
    pkl_fname = f"{export_model_fname}/tY_{args.data_sample}.pkl"
    pickle.dump(y_scaler, open(pkl_fname, "wb"))
    print(f"[INFO] Output variables scaler correctly saved to '{pkl_fname}'")
