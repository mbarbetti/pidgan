from argparse import ArgumentParser

MODELS = ["Rich", "Muon", "GlobalPID", "GlobalMuonId", "isMuon"]
PARTICLES = ["muon", "pion", "kaon", "proton"]
LABELS = ["sim", "calib"]


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
        "-l",
        "--label",
        required=True,
        choices=LABELS,
        help="prepare dataset from simulated/calibration samples",
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
        help="maximum number of instancens downloaded from the overall files (default:-1)",
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


def argparser_training(description=None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-p",
        "--particle",
        required=True,
        choices=PARTICLES,
        help="train the Rich model for the selected long-lived charged particle",
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
        "--fullsim",
        action="store_true",
        help="train the Rich model using data taken from simulated samples (default: True)",
    )
    parser.add_argument(
        "--calibration",
        dest="fullsim",
        action="store_false",
        help="train the Rich model using data taken from calibration samples (default: False)",
    )
    parser.set_defaults(fullsim=True)
    parser.add_argument(
        "--weights",
        action="store_true",
        help="train the Rich model using weights when available (default: True)",
    )
    parser.add_argument("--no-weights", dest="weights", action="store_false")
    parser.set_defaults(weights=True)
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
