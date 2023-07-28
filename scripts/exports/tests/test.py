import ctypes
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from utils import FullPipe, GanPipe, isMuonPipe

from pidgan.utils.preprocessing import invertColumnTransformer

MODELS = ["Rich", "Muon", "GlobalPID", "GlobalMuonId"]
PARTICLES = ["muon", "pion", "kaon", "proton"]
DATA_SAMPLES = [f"{s}-2016MU" for s in ["sim9", "sim10", "calib"]]

DTYPE = np.float32
BATCH_SIZE = 512

LATENT_DIM = 64
N_OUTPUT_ISMUON = 1
N_OUTPUT_RICH = 4
N_OUTPUT_MUON = 2
N_OUTPUT_GLOBALPID = 7
N_OUTPUT_GLOBALMUONID = 2
N_OUTPUT_FULL = (
    N_OUTPUT_RICH + N_OUTPUT_MUON + N_OUTPUT_GLOBALPID + N_OUTPUT_GLOBALMUONID
)

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
    "--max_rel_err",
    default=0.001,
    help="maximum relative error allowed for Python-C conversion (default: 0.001)",
)
parser.add_argument(
    "-P",
    "--min_percentage",
    default=95.0,
    help="minimum percentage of well-converted instances (default: 95.0)",
)
args = parser.parse_args()

max_rel_err = float(args.max_rel_err)
min_percentage = float(args.min_percentage)

# +-------------------+
# |   Initial setup   |
# +-------------------+

with open("../config/directories.yml") as file:
    config_dir = yaml.full_load(file)

data_dir = config_dir["data_dir"]
models_dir = config_dir["models_dir"]

chunk_size = int(args.chunk_size)
label = f"{args.shared_object}".split(".")[0].split("_")[-1]

# +------------------+
# |   Data loading   |
# +------------------+

data_model = MODELS[0]

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

x = npzfile["x"].astype(DTYPE)[:chunk_size]
x = invertColumnTransformer(x_scaler, x)
rnd_noise = np.random.uniform(0.0, 1.0, size=(len(x), LATENT_DIM * 4))

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
float_p = ctypes.POINTER(ctypes.c_float)

ismuon_pyout = ismuon_pipe.predict_proba(x, batch_size=BATCH_SIZE)

rel_errs = list()
for x_row, pyout_row in tqdm(
    zip(x, ismuon_pyout),
    desc="Testing isMuon consistency (C vs. Python)",
    total=len(x),
    unit="data",
):
    in_f = x_row.astype(ctypes.c_float)
    out_f = np.empty(N_OUTPUT_ISMUON, dtype=ctypes.c_float)
    getattr(dll, f"isMuon_{args.particle}_pipe")(
        out_f.ctypes.data_as(float_p), in_f.ctypes.data_as(float_p)
    )
    if pyout_row != 0.0:
        rel_errs.append(abs((out_f - pyout_row) / pyout_row))

print(pd.DataFrame(rel_errs, columns=["isMuon_err"]).describe())
counts = np.count_nonzero(np.array(rel_errs) < max_rel_err)
percentage = 100 * float(counts / (1 + len(rel_errs)))
if percentage < min_percentage:
    raise Exception(
        f"C and Python isMuonANN implementations were found inconsistent ({100.0 - percentage:.2f}%)"
    )

mu_rnd = np.random.uniform(0.0, 1.0, size=(len(x), 1))
ismuon_flag = np.where(mu_rnd < ismuon_pyout, 1.0, 0.0)

full_pyout = full_pipe.predict(x, rnd_noise, ismuon_flag, batch_size=BATCH_SIZE)

rel_errs = dict()
for model in MODELS:
    rel_errs[model] = list()

for x_row, rnd_row, ismuon_row, pyout_row in tqdm(
    zip(x, rnd_noise, ismuon_flag, full_pyout),
    desc="Testing GANs consistency (C vs. Python)",
    total=len(x),
    unit="data",
):
    in_f = np.hstack((x_row, ismuon_row)).astype(ctypes.c_float)
    out_f = np.empty(N_OUTPUT_FULL, dtype=ctypes.c_float)
    rnd_f = rnd_row.astype(ctypes.c_float)
    getattr(dll, f"full_{args.particle}_pipe")(
        out_f.ctypes.data_as(float_p),
        in_f.ctypes.data_as(float_p),
        rnd_f.ctypes.data_as(float_p),
    )
    i = 0
    for model, n_out in zip(
        MODELS,
        [N_OUTPUT_RICH, N_OUTPUT_MUON, N_OUTPUT_GLOBALPID, N_OUTPUT_GLOBALMUONID],
    ):
        if np.all(pyout_row[i : (i + n_out)]) != 0.0:
            rel_errs[model].append(
                list(
                    np.abs(
                        (out_f[i : (i + n_out)] - pyout_row[i : (i + n_out)])
                        / pyout_row[i : (i + n_out)]
                    ).flatten()
                )
            )
        i += n_out

for model in MODELS:
    rel_errs_ = np.array(rel_errs[model])
    print(
        pd.DataFrame(
            rel_errs_, columns=[f"{model}#{i}_err" for i in range(rel_errs_.shape[1])]
        ).describe()
    )
    counts = np.count_nonzero(rel_errs_ < max_rel_err, axis=0)
    percentage = 100.0 * (counts / (1.0 + len(rel_errs_)))
    if percentage.min() < min_percentage:
        raise Exception(
            f"C and Python {model}GAN implementations were found inconsistent ({100.0 - percentage.min():.2f}%)"
        )
