import socket
from argparse import ArgumentParser

MODELS = ["Rich", "Muon", "GlobalPID", "GlobalMuonId", "isMuon"]
PARTICLES = ["muon", "pion", "kaon", "proton"]
DATA_SAMPLES = [f"{s}-2016MU" for s in ["sim9", "sim10", "calib"]]
ADDRESS = socket.gethostbyname(socket.gethostname())


def argparser_preprocessing(description=None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=MODELS,
        help="prepare dataset for the selected LHCb sub-detector",
    )
    parser.add_argument(
        "-p",
        "--particle",
        required=True,
        choices=PARTICLES,
        help="prepare dataset for the selected long-lived charged particle",
    )
    parser.add_argument(
        "-F",
        "--filename",
        default="./data/LamarrPidTraining.root",
        help="path of the files from which downoloading data (default: './data/LamarrPidTraining.root')",
    )
    parser.add_argument(
        "-M",
        "--max_files",
        default=10,
        help="maximum number of files from which downloading data (default: 10)",
    )
    parser.add_argument(
        "-C",
        "--chunk_size",
        default=-1,
        help="maximum number of instancens downloaded from the overall files (default: -1)",
    )
    parser.add_argument(
        "-D",
        "--data_sample",
        required=True,
        choices=DATA_SAMPLES,
        help="prepare dataset from simulated/calibration samples",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="take also the weights from the data files (default: False)",
    )
    parser.add_argument("--no-weights", dest="weights", action="store_false")
    parser.set_defaults(weights=False)
    parser.add_argument(
        "--tricks",
        action="store_true",
        help="combine smartly some of the varibles downloaded from the data files (default: False)",
    )
    parser.add_argument("--no-tricks", dest="tricks", action="store_false")
    parser.set_defaults(tricks=False)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print various pandas' DataFrames once created (default: False)",
    )
    parser.add_argument("--no-verbose", dest="saving", action="store_false")
    parser.set_defaults(verbose=False)
    return parser


def argparser_training(model, description=None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-p",
        "--particle",
        required=True,
        choices=PARTICLES,
        help=f"train the {model} model for the selected long-lived charged particle",
    )
    parser.add_argument(
        "-E",
        "--num_epochs",
        default=100,
        help="number of epochs to be used for the training procedure (default: 100)",
    )
    parser.add_argument(
        "-C",
        "--chunk_size",
        default=-1,
        help="maximum number of instancens to be used for training/validation (default: -1)",
    )
    parser.add_argument(
        "-T",
        "--train_ratio",
        default=0.7,
        help="fraction of instances to be used for training (default: 0.7)",
    )
    parser.add_argument(
        "-D",
        "--data_sample",
        required=True,
        choices=DATA_SAMPLES,
        help="prepare dataset from simulated/calibration samples",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help=f"train the {model} model using weights when available (default: False)",
    )
    parser.add_argument("--no-weights", dest="weights", action="store_false")
    parser.set_defaults(weights=False)
    parser.add_argument(
        "--test",
        action="store_true",
        help="enable overwriting for model, images and reports since test execution (default: False)",
    )
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.set_defaults(test=False)
    parser.add_argument(
        "--saving",
        action="store_true",
        help="enable to save the trained model and all the images produced (default: False)",
    )
    parser.add_argument("--no-saving", dest="saving", action="store_false")
    parser.set_defaults(saving=False)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="enable overwriting for model, images and reports since latest execution (default: False)",
    )
    parser.add_argument("--no-latest", dest="latest", action="store_false")
    parser.set_defaults(latest=False)
    return parser


def argparser_optimization(description=None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=MODELS,
        help="optimize the GAN model for the selected LHCb sub-detector",
    )
    parser.add_argument(
        "-p",
        "--particle",
        required=True,
        choices=PARTICLES,
        help="optimize the GAN model for the selected long-lived charged particle",
    )
    parser.add_argument(
        "-n",
        "--node_name",
        default=f"{ADDRESS}",
        help="name given to the computing node that runs the optimization study (default: IP address)",
    )
    parser.add_argument(
        "-D",
        "--data_sample",
        required=True,
        choices=DATA_SAMPLES,
        help="prepare dataset from simulated/calibration samples",
    )
    parser.add_argument(
        "-T",
        "--train_ratio",
        default=0.7,
        help="fraction of instances to be used for a trial of training (default: 0.7)",
    )
    parser.add_argument(
        "-S",
        "--min_score_for_report",
        default=0.1,
        help="minimum optimization score to produce the HTML report (default: 0.1)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="ensure that the optimization study is running on GPU (default: False)",
    )
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    parser.add_argument(
        "--weights",
        action="store_true",
        help="optimize the GAN model using weights when available (default: False)",
    )
    parser.add_argument("--no-weights", dest="weights", action="store_false")
    parser.set_defaults(weights=False)
    parser.add_argument(
        "--test",
        action="store_true",
        help="enable overwriting for model, images and reports since test execution (default: False)",
    )
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.set_defaults(test=False)
    parser.add_argument(
        "--saving",
        action="store_true",
        help="enable to save the trained model and all the images produced (default: False)",
    )
    parser.add_argument("--no-saving", dest="saving", action="store_false")
    parser.set_defaults(saving=False)
    return parser
