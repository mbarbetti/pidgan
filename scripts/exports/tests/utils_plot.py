import matplotlib.pyplot as plt
import numpy as np

out_vars = {
    "isMuon": ["isMuon"],
    "Rich": ["RichDLLe", "RichDLLmu", "RichDLLk", "RichDLLp"],
    "Muon": ["MuonMuLL", "MuonBgLL"],
    "GlobalPIDmu": [
        "PIDe",
        "PIDK",
        "PIDp",
        "ProbNNe",
        "ProbNNpi",
        "ProbNNk",
        "ProbNNp",
        "PIDmu",
        "ProbNNmu",
    ],
}


def validation_histogram(
    model_name, particle, py_out, c_out, log_scale=False, export_dirname="./images"
) -> None:
    var_names = out_vars[model_name]

    if model_name == "isMuon":
        title = model_name
        plt.figure(figsize=(8, 5))
        nrows, ncols = (1, 1)
    elif model_name in ["Rich", "Muon"]:
        title = model_name
        plt.figure(figsize=(8 * len(var_names), 5))
        nrows, ncols = (1, len(var_names))
    elif model_name == "GlobalPIDmu":
        title = model_name[:-2]
        plt.figure(figsize=(24, 15))
        nrows, ncols = (3, 3)

    export_fname = f"{export_dirname}/{title}-{particle}-hist"
    if log_scale:
        export_fname += "-log"

    i = 0
    for iRow in range(nrows):
        for iCol in range(ncols):
            plt.subplot(nrows, ncols, i + 1)
            plt.xlabel(var_names[i], fontsize=12)
            plt.ylabel("Candidates", fontsize=12)
            _, bins_, _ = plt.hist(
                py_out[:, i], bins=100, color="#3288bd", label="Original"
            )
            plt.hist(
                c_out[:, i],
                bins=bins_,
                histtype="step",
                color="#fc8d59",
                lw=2,
                label="Deployed",
            )

            if log_scale:
                plt.yscale("log")
            plt.legend(loc=None, fontsize=10)

            i += 1

    plt.savefig(f"{export_fname}.png", dpi=300)
    print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    plt.close()
