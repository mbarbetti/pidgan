import pickle
from argparse import ArgumentParser

import scikinC
from tensorflow import keras

MODELS = ["Rich", "Muon", "GlobalPIDmu", "GlobalPIDh"]
PARTICLES = ["muon", "pion", "kaon", "proton"]
DATA_SAMPLES = [f"{s}-2016MU" for s in ["sim9", "sim10", "calib"]]

# +------------------+
# |   Parser setup   |
# +------------------+

parser = ArgumentParser(description="GAN models deployment")
parser.add_argument(
    "-m",
    "--model",
    required=True,
    choices=MODELS,
    help="LHCb sub-detector parameterized by the passed model directory",
)
parser.add_argument(
    "-p",
    "--particle",
    required=True,
    choices=PARTICLES,
    help="long-lived charged particle parameterized by the passed model directory",
)
parser.add_argument(
    "-D",
    "--data_sample",
    required=True,
    choices=DATA_SAMPLES,
    help="data samples used to train the parameterizations provided within model directory",
)
parser.add_argument(
    "-M",
    "--model_dir",
    required=True,
    help="path of the directory containing the parameterizations to be deployed",
)
args = parser.parse_args()

# +-------------------+
# |   Model loading   |
# +-------------------+

with open(f"{args.model_dir}/tX.pkl", "rb") as file:
    x_scaler = pickle.load(file)

generator = keras.models.load_model(f"{args.model_dir}/saved_generator")

with open(f"{args.model_dir}/tY.pkl", "rb") as file:
    y_scaler = pickle.load(file)

# +---------------------------+
# |   scikinC transpilation   |
# +---------------------------+

models_to_deploy = dict()
models_to_deploy[f"{args.model}_{args.particle}_tX"] = x_scaler
models_to_deploy[f"{args.model}_{args.particle}_dnn"] = generator
models_to_deploy[f"{args.model}_{args.particle}_tY"] = y_scaler

label = f"{args.model_dir}".split("/")[-1].split("_")[0]
c_fname = f"{args.model}-{args.particle}_{args.data_sample}_{label}-gan.C"

print(
    scikinC.convert(models_to_deploy, float_t="float"),
    file=open(f"/tmp/{c_fname}", "w"),
)
