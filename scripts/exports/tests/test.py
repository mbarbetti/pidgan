import ctypes
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml
from scipy.stats import percentileofscore as pos
from tqdm import tqdm
from utils_pipe import FullPipe, GanPipe, isMuonPipe
from utils_plot import out_vars, validation_histogram

from pidgan.utils.preprocessing import invertColumnTransformer

MODELS = ["Rich", "Muon", "GlobalPIDmu", "GlobalPIDh"]
PARTICLES = ["muon", "pion", "kaon", "proton"]
DATA_SAMPLES = [f"{s}-2016MU" for s in ["sim9", "sim10", "calib"]]

DTYPE = np.float32
CTYPE = ctypes.c_float
BATCH_SIZE = 512

LATENT_DIM = 64
N_OUTPUT_ISMUON = 1
N_OUTPUT_RICH = 4
N_OUTPUT_MUON = 2
N_OUTPUT_GLOBALPID_MU = 9
N_OUTPUT_GLOBALPID_HAD = 8
N_OUTPUT_FULL = N_OUTPUT_RICH + N_OUTPUT_MUON + N_OUTPUT_GLOBALPID_MU

MAX_ABS_ERR = 1e-3

# +------------------+
# |   Parser setup   |
# +------------------+

parser = ArgumentParser(description="Deployment validation")
parser.add_argument(
    "-o",
    "--shared_object",
    required=True,
    help="path of the shared object that contains the compiled models",
)
parser.add_argument(
    "-p",
    "--particle",
    required=True,
    choices=PARTICLES,
    help="select models according to the passed long-lived charged particle",
)
parser.add_argument(
    "-C",
    "--chunk_size",
    default=10_000,
    help="maximum number of instancens to be used for validation (default: 10_000)",
)
parser.add_argument(
    "-D",
    "--data_sample",
    required=True,
    choices=DATA_SAMPLES,
    help="prepare dataset from simulated/calibration samples",
)
parser.add_argument(
    "-E",
    "--max_q_err",
    default=0.001,
    help="maximum quantile error allowed for Python-C conversion (default: 0.001)",
)
parser.add_argument(
    "-P",
    "--err_patience",
    default=1.0,
    help="maximum percentage of wrong-converted instances (default: 1.0)",
)
parser.add_argument(
    "--figure",
    action="store_true",
    help="save figure of original VS. deployed histograms (default: True)",
)
parser.add_argument("--no-figure", dest="saving", action="store_false")
parser.set_defaults(figure=True)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="print various pandas' DataFrames for debugging (default: False)",
)
parser.add_argument("--no-verbose", dest="saving", action="store_false")
parser.set_defaults(verbose=False)
args = parser.parse_args()

# +-------------------+
# |   Initial setup   |
# +-------------------+

with open("../config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
models_dir = config_dir["models_dir"]

chunk_size = int(args.chunk_size)
max_q_err = float(args.max_q_err)
err_patience = float(args.err_patience) / 1e2
label = f"{args.shared_object}".split(".")[0].split("_")[-1]

# +------------------+
# |   Data loading   |
# +------------------+

data_model = MODELS[0]  # only interested in Python VS. C consistency

npzfile = np.load(
    f"{data_dir}/pidgan-{data_model}-{args.particle}-{args.data_sample}-data.npz"
)

filepath = os.path.join(
    models_dir,
    f"{data_model}_{args.particle}_models",
    f"{label}_{data_model}GAN-{args.particle}_{args.data_sample}_model",
)
with open(f"{filepath}/tX.pkl", "rb") as file:
    x_scaler = pickle.load(file)

x_prep = npzfile["x"].astype(DTYPE)[:chunk_size]
x = invertColumnTransformer(x_scaler, x_prep)
rnd_noise = np.random.normal(0.0, 1.0, size=(len(x), LATENT_DIM * 4)).astype(DTYPE)

if args.verbose:
    print(pd.DataFrame(x, columns=["P", "ETA", "nTracks", "trackcharge"]).describe())

# +--------------------------+
# |   Python full pipeline   |
# +--------------------------+

filepath = os.path.join(
    models_dir,
    f"isMuon_{args.particle}_models",
    f"{label}_isMuonANN-{args.particle}_{args.data_sample}_model",
)
ismuon_pipe = isMuonPipe(filepath=filepath)

ml_pipes = dict()
for model in MODELS:
    filepath = os.path.join(
        models_dir,
        f"{model}_{args.particle}_models",
        f"{label}_{model}GAN-{args.particle}_{args.data_sample}_model",
    )
    ml_pipes[model] = GanPipe(filepath=filepath)

full_pipe = FullPipe(ml_pipes=ml_pipes)

# +---------------------+
# |   C full pipeline   |
# +---------------------+

dll = ctypes.CDLL(args.shared_object)
float_p = ctypes.POINTER(CTYPE)

ismuon_c = list()
for x_row in tqdm(
    x, desc="Testing isMuon consistency (C vs. Python)", total=len(x), unit="data"
):
    in_f = x_row.astype(CTYPE)
    out_f = np.empty(N_OUTPUT_ISMUON, dtype=CTYPE)
    getattr(dll, f"isMuon_{args.particle}_pipe")(
        out_f.ctypes.data_as(float_p), in_f.ctypes.data_as(float_p)
    )
    ismuon_c.append(out_f)

ismuon_py = ismuon_pipe.predict_proba(x, batch_size=BATCH_SIZE).flatten()
ismuon_c = np.array(ismuon_c).flatten()

mu_rnd = np.random.uniform(0.0, 1.0, size=(len(x), 1))
ismuon_flag = np.where(mu_rnd < ismuon_py[:, None], 1.0, 0.0)

full_c = list()
for x_row, rnd_row, ismuon_row in tqdm(
    zip(x, rnd_noise, ismuon_flag),
    desc="Testing GANs consistency (C vs. Python)",
    total=len(x),
    unit="data",
):
    in_f = np.hstack((x_row, ismuon_row)).astype(CTYPE)
    out_f = np.empty(N_OUTPUT_FULL, dtype=CTYPE)
    rnd_f = rnd_row.astype(CTYPE)
    getattr(dll, f"full_{args.particle}_pipe")(
        out_f.ctypes.data_as(float_p),
        in_f.ctypes.data_as(float_p),
        rnd_f.ctypes.data_as(float_p),
    )
    full_c.append(out_f)

full_py = full_pipe.predict(x, rnd_noise, ismuon_flag, batch_size=BATCH_SIZE)
full_c = np.array(full_c)

# +-----------------------------+
# |   isMuon pipe consistency   |
# +-----------------------------+

ismuon_err_df = pd.DataFrame()
ismuon_q_err_py = pos(ismuon_py, ismuon_py)
ismuon_q_err_c = pos(ismuon_py, ismuon_c)
ismuon_err_df["q_err_isMuon"] = (ismuon_q_err_py - ismuon_q_err_c) / 1e2
ismuon_err_df["abs_err_isMuon"] = abs(ismuon_py - ismuon_c)

err_counts = np.count_nonzero(
    (abs(ismuon_err_df["q_err_isMuon"]) > max_q_err)
    & (ismuon_err_df["abs_err_isMuon"] > MAX_ABS_ERR)
)
err_percentage = err_counts / len(ismuon_err_df["q_err_isMuon"])
if err_percentage > err_patience:
    print("\n*** isMuon errors ***")
    print(ismuon_err_df[["q_err_isMuon", "abs_err_isMuon"]].describe(), "\n")
    print("*** Number error instances ***")
    print(
        f"abs(q_err_isMuon) > {max_q_err} & abs_err_isMuon > {MAX_ABS_ERR} : {err_counts} / {len(ismuon_err_df['q_err_isMuon'])}\n"
    )
    raise Exception(
        f"C and Python isMuon model implementations were found inconsistent ({100 * err_percentage:.2f}%)"
    )

if args.verbose:
    print(ismuon_err_df["q_err_isMuon"].describe())
    print(ismuon_err_df["abs_err_isMuon"].describe())

if args.figure:
    for log_scale in [False, True]:
        validation_histogram(
            model_name="isMuon",
            particle=args.particle,
            py_out=ismuon_py[:, None],
            c_out=ismuon_c[:, None],
            log_scale=log_scale,
            export_dirname="./images",
        )

# +---------------------------+
# |   Full pipe consistency   |
# +---------------------------+

i = 0
for model, n_out in zip(
    MODELS[:-1], [N_OUTPUT_RICH, N_OUTPUT_MUON, N_OUTPUT_GLOBALPID_MU]
):
    model_err_df = pd.DataFrame()
    for j in range(n_out):
        model_q_err_py = pos(full_py[:, i + j], full_py[:, i + j])
        model_q_err_c = pos(full_py[:, i + j], full_c[:, i + j])
        model_err_df[f"q_err_{out_vars[model][j]}"] = (
            model_q_err_py - model_q_err_c
        ) / 1e2
        model_err_df[f"abs_err_{out_vars[model][j]}"] = abs(
            full_py[:, i + j] - full_c[:, i + j]
        )

        err_counts = np.count_nonzero(
            (abs(model_err_df[f"q_err_{out_vars[model][j]}"]) > max_q_err)
            & (model_err_df[f"abs_err_{out_vars[model][j]}"] > MAX_ABS_ERR)
        )
        err_percentage = err_counts / len(model_err_df[f"q_err_{out_vars[model][j]}"])
        if err_percentage > err_patience:
            print(f"\n*** {out_vars[model][j]} errors ***")
            print(
                model_err_df[
                    [f"q_err_{out_vars[model][j]}", f"abs_err_{out_vars[model][j]}"]
                ].describe(),
                "\n",
            )
            print("*** Number error instances ***")
            print(
                f"abs(q_err_{out_vars[model][j]}) > {max_q_err} & abs_err_{out_vars[model][j]} > {MAX_ABS_ERR} : {err_counts} / {len(model_err_df[f'q_err_{out_vars[model][j]}'])}\n"
            )
            raise Exception(
                f"C and Python {out_vars[model][j]} model implementations were found inconsistent ({100 * err_percentage:.2f}%)"
            )

    if args.verbose:
        print(
            model_err_df[
                [f"q_err_{out_vars[model][j]}" for j in range(n_out)]
            ].describe()
        )
        print(
            model_err_df[
                [f"abs_err_{out_vars[model][j]}" for j in range(n_out)]
            ].describe()
        )

    if args.figure:
        for log_scale in [False, True]:
            validation_histogram(
                model_name=model,
                particle=args.particle,
                py_out=full_py[:, i : (i + n_out)],
                c_out=full_c[:, i : (i + n_out)],
                log_scale=log_scale,
                export_dirname="./images",
            )

    i += n_out
