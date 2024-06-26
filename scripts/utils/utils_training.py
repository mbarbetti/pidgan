import socket
import pidgan
import keras as k
import numpy as np
import tensorflow as tf

from html_reports import Report
from pidgan.utils.reports import getSummaryHTML

from .utils_plot import (
    binned_validation_histogram,
    correlation_histogram,
    identification_efficiency,
    learn_rate_scheduling,
    learning_curves,
    metric_curves,
    selection_efficiency,
    validation_2d_histogram,
    validation_histogram,
)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "bce": "Binary cross-entropy",
    "js_div": "Jensen-Shannon divergence",
    "kl_div": "Kullback-Leibler divergence",
    "mae": "Mean Absolute Error",
    "mse": "Mean Squared Error",
    "rmse": "Root Mean Squared Error",
    "wass_dist": "Wasserstein distance",
}


def fill_html_report(
    report,
    title,
    train_duration,
    report_datetime,
    particle,
    data_sample,
    trained_with_weights,
    hp_dict,
    model_arch,
    model_labels,
) -> Report:
    report.add_markdown(f'<h1 align="center">{title}</h1>')

    ## General information
    date, hour = report_datetime
    info = [
        f"- Script executed on **{socket.gethostname()}**",
        f"- Model training completed in **{train_duration}**",
        f"- Model training executed with **TF {tf.__version__}** "
        f"(Keras {k.__version__}) and **pidgan {pidgan.__version__}**",
        f"- Report generated on **{date}** at **{hour}**",
        f"- Model trained on **{particle}** tracks",
    ]

    if "calib" not in data_sample:
        info += [f"- Model trained on **detailed simulated** samples ({data_sample})"]
    else:
        info += [f"- Model trained on **calibration** samples ({data_sample})"]
        if trained_with_weights:
            info += ["- Any background components subtracted using **sWeights**"]
        else:
            info += ["- **sWeights not applied**"]

    report.add_markdown("\n".join([i for i in info]))
    report.add_markdown("---")

    ## Hyperparameters and other details
    report.add_markdown('<h2 align="center">Hyperparameters and other details</h2>')
    hyperparams = ""
    for key, val in hp_dict.items():
        hyperparams += f"- **{key}:** {val}\n"
    report.add_markdown(hyperparams)
    report.add_markdown("---")

    ## Models architecture
    for model, label in zip(model_arch, model_labels):
        if model is not None:
            report.add_markdown(f'<h2 align="center">{label} architecture</h2>')
            report.add_markdown(f"**Model name:** {model.name}")
            html_table, params_details = getSummaryHTML(model.plain_keras)
            model_weights = ""
            for key, num in zip(
                ["Total", "Trainable", "Non-trainable"], params_details
            ):
                model_weights += f"- **{key} params:** {num}\n"
            report.add_markdown(html_table)
            report.add_markdown(model_weights)
            report.add_markdown("---")

    return report


def prepare_training_plots(
    report,
    model,
    history,
    metrics=None,
    num_epochs=None,
    loss_name=None,
    from_validation_set=True,
    referee_available=False,
    save_images=False,
    images_dirname="./images",
) -> None:
    report.add_markdown('<h2 align="center">Training plots</h2>')

    if num_epochs is None:
        key = list(history.keys())[0]
        num_epochs = len(history[key])

    start_epoch = int(num_epochs / 20)

    if model != "isMuon":
        learning_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            keys=["g_loss", "d_loss"]
            if not referee_available
            else ["g_loss", "d_loss", "r_loss"],
            colors=["#3288bd", "#fc8d59"]
            if not referee_available
            else ["#3288bd", "#fc8d59", "#4dac26"],
            labels=["generator", "discriminator"]
            if not referee_available
            else ["generator", "discriminator", "referee"],
            legend_loc=None,
            save_figure=save_images,
            scale_curves=True,
            export_fname=f"{images_dirname}/learn-curves",
        )

        learn_rate_scheduling(
            report=report,
            history=history,
            start_epoch=0,
            keys=["g_lr", "d_lr"]
            if not referee_available
            else ["g_lr", "d_lr", "r_lr"],
            colors=["#3288bd", "#fc8d59"]
            if not referee_available
            else ["#3288bd", "#fc8d59", "#4dac26"],
            labels=["generator", "discriminator"]
            if not referee_available
            else ["generator", "discriminator", "referee"],
            legend_loc="upper right",
            save_figure=save_images,
            export_fname=f"{images_dirname}/lr-sched",
        )

        metric_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            key="g_loss",
            ylabel=loss_name,
            title="Generator learning curves",
            validation_set=from_validation_set,
            colors=["#d01c8b", "#4dac26"],
            labels=["training set", "validation set"],
            legend_loc=None,
            yscale="linear",
            save_figure=save_images,
            export_fname=f"{images_dirname}/gen-loss",
        )

        metric_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            key="d_loss",
            ylabel=loss_name,
            title="Discriminator learning curves",
            validation_set=from_validation_set,
            colors=["#d01c8b", "#4dac26"],
            labels=["training set", "validation set"],
            legend_loc=None,
            yscale="linear",
            save_figure=save_images,
            export_fname=f"{images_dirname}/disc-loss",
        )

        if referee_available:
            metric_curves(
                report=report,
                history=history,
                start_epoch=start_epoch,
                key="r_loss",
                ylabel="Binary cross-entropy",
                title="Referee learning curves",
                validation_set=from_validation_set,
                colors=["#d01c8b", "#4dac26"],
                labels=["training set", "validation set"],
                legend_loc=None,
                yscale="linear",
                save_figure=save_images,
                export_fname=f"{images_dirname}/ref-loss",
            )

    else:
        metric_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            key="loss",
            ylabel="Binary cross-entropy",
            title="Learning curves",
            validation_set=from_validation_set,
            colors=["#d01c8b", "#4dac26"],
            labels=["training set", "validation set"],
            legend_loc=None,
            yscale="linear",
            save_figure=save_images,
            export_fname=f"{images_dirname}/bce-loss",
        )

        learn_rate_scheduling(
            report=report,
            history=history,
            start_epoch=0,
            keys=["lr"],
            colors=["#3288bd"],
            labels=["classifier"],
            legend_loc="upper right",
            save_figure=save_images,
            export_fname=f"{images_dirname}/lr-sched",
        )

    if metrics is not None:
        for metric in metrics:
            metric_curves(
                report=report,
                history=history,
                start_epoch=start_epoch,
                key=metric,
                ylabel=METRIC_LABELS[metric],
                title="Metric curves",
                validation_set=from_validation_set,
                colors=["#d01c8b", "#4dac26"],
                labels=["training set", "validation set"],
                legend_loc=None,
                yscale="linear",
                save_figure=save_images,
                export_fname=f"{images_dirname}/{metric}-curves",
            )

    report.add_markdown("---")


def prepare_validation_plots(
    report,
    model,
    x_true,
    y_true,
    y_pred,
    y_vars,
    weights=None,
    from_fullsim=True,
    save_images=False,
    images_dirname="./images",
) -> None:
    if model != "isMuon":
        if model == "Rich":
            for rich_dll in [
                ["RichDLLmu", "RichDLLe", "RichDLLmue"],
                ["RichDLLp", "RichDLLk", "RichDLLpk"],
            ]:
                idx_0 = y_vars.index(rich_dll[0])
                idx_1 = y_vars.index(rich_dll[1])
                new_y_true = y_true[:, idx_0] - y_true[:, idx_1]
                y_true = np.c_[y_true, new_y_true]
                new_y_pred = y_pred[:, idx_0] - y_pred[:, idx_1]
                y_pred = np.c_[y_pred, new_y_pred]
                y_vars += [rich_dll[2]]
        elif model == "Muon":
            idx_mu = y_vars.index("MuonMuLL")
            idx_bg = y_vars.index("MuonBgLL")
            new_y_true = y_true[:, idx_mu] - y_true[:, idx_bg]
            y_true = np.c_[y_true, new_y_true]
            new_y_pred = y_pred[:, idx_mu] - y_pred[:, idx_bg]
            y_pred = np.c_[y_pred, new_y_pred]
            y_vars += ["muDLL"]
        elif "GlobalPID" in model:
            for comb_dll in [["PIDmu", "PIDe", "PIDmue"], ["PIDp", "PIDK", "PIDpK"]]:
                idx_0 = y_vars.index(comb_dll[0])
                idx_1 = y_vars.index(comb_dll[1])
                new_y_true = y_true[:, idx_0] - y_true[:, idx_1]
                y_true = np.c_[y_true, new_y_true]
                new_y_pred = y_pred[:, idx_0] - y_pred[:, idx_1]
                y_pred = np.c_[y_pred, new_y_pred]
                y_vars += [comb_dll[2]]

        for i, y_var in enumerate(y_vars):
            report.add_markdown(f'<h2 align="center">Validation plots of {y_var}</h2>')

            for log_scale in [False, True]:
                validation_histogram(
                    report=report,
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}-hist",
                )

            for log_scale in [False, True]:
                correlation_histogram(
                    report=report,
                    data_corr=x_true[:, 0],
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    range_corr=[3.0, 153.0],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel="Momentum [GeV/$c$]",
                    ylabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}_vs_p-corr_hist",
                )

            for log_scale in [False, True]:
                binned_validation_histogram(
                    report=report,
                    data_bin=x_true[:, 0],
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    boundaries_bin=[3.0, 10.0, 25.0, 50.0, 150.0],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    symbol_bin="$p$",
                    unit_bin="[GeV/$c$]",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}-hist",
                )

            selection_efficiency(
                report=report,
                data_bin=x_true[:, 0],
                data_ref=y_true[:, i],
                data_gen=y_pred[:, i],
                range_bin=[3.0, 153.0],
                weights_ref=weights,
                weights_gen=weights,
                title=f"{y_var}",
                xlabel="Momentum [GeV/$c$]",
                label_ref="Full simulation" if from_fullsim else "Calibration samples",
                label_gen="GAN-based model",
                save_figure=save_images,
                export_fname=f"{images_dirname}/{y_var}_vs_p-eff",
            )

            for log_scale in [False, True]:
                correlation_histogram(
                    report=report,
                    data_corr=x_true[:, 1],
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    range_corr=[1.5, 5.5],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel="Pseudorapidity",
                    ylabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}_vs_eta-corr_hist",
                )

            for log_scale in [False, True]:
                binned_validation_histogram(
                    report=report,
                    data_bin=x_true[:, 1],
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    boundaries_bin=[1.5, 2.5, 3.5, 4.5, 5.5],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    symbol_bin="$\eta$",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}-hist",
                )

            selection_efficiency(
                report=report,
                data_bin=x_true[:, 1],
                data_ref=y_true[:, i],
                data_gen=y_pred[:, i],
                range_bin=[1.5, 5.5],
                weights_ref=weights,
                weights_gen=weights,
                title=f"{y_var}",
                xlabel="Pseudorapidity",
                label_ref="Full simulation" if from_fullsim else "Calibration samples",
                label_gen="GAN-based model",
                save_figure=save_images,
                export_fname=f"{images_dirname}/{y_var}_vs_eta-eff",
            )

            for log_scale in [False, True]:
                correlation_histogram(
                    report=report,
                    data_corr=x_true[:, 2],
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    range_corr=[0, 800],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel="$\mathtt{nTracks}$",
                    ylabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}_vs_nTracks-corr_hist",
                )

            for log_scale in [False, True]:
                binned_validation_histogram(
                    report=report,
                    data_bin=x_true[:, 2],
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    boundaries_bin=[0, 100, 250, 350, 800],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    symbol_bin="$\mathtt{nTracks}$",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}-hist",
                )

            selection_efficiency(
                report=report,
                data_bin=x_true[:, 2],
                data_ref=y_true[:, i],
                data_gen=y_pred[:, i],
                range_bin=[0, 800],
                weights_ref=weights,
                weights_gen=weights,
                title=f"{y_var}",
                xlabel="$\mathtt{nTracks}$",
                label_ref="Full simulation" if from_fullsim else "Calibration samples",
                label_gen="GAN-based model",
                save_figure=save_images,
                export_fname=f"{images_dirname}/{y_var}_vs_nTracks-eff",
            )

            report.add_markdown("---")

    else:
        report.add_markdown('<h2 align="center">Validation plots</h2>')

        for log_scale in [False, True]:
            validation_2d_histogram(
                report=report,
                x_true=x_true[:, 0],
                y_true=x_true[:, 1],
                range_x=[3.0, 153.0],
                range_y=[1.5, 5.5],
                weights_true=weights * y_true if weights else y_true,
                weights_pred=weights * y_pred if weights else y_pred,
                xlabel="Momentum [GeV/$c$]",
                ylabel="Pseudorapidity",
                label_true="Full simulation" if from_fullsim else "Calibration samples",
                label_pred="ANN-based model",
                log_scale=log_scale,
                save_figure=save_images,
                export_fname=f"{images_dirname}/eta_vs_p-isMuon-hist",
            )

        identification_efficiency(
            report=report,
            x_true=x_true[:, 0],
            id_true=y_true,
            id_pred=y_pred,
            data_bin=x_true[:, 1],
            range_true=[3.0, 153.0],
            boundaries_bin=[1.5, 2.5, 3.5, 4.5, 5.5],
            weights_true=weights,
            weights_pred=weights,
            xlabel="Momentum [GeV/$c$]",
            label_model="isMuon",
            label_true="Full simulation" if from_fullsim else "Calibration samples",
            label_pred="ANN-based model",
            symbol_bin="$\eta$",
            save_figure=save_images,
            export_fname=f"{images_dirname}/eta_vs_p-isMuon-eff",
        )

        for log_scale in [False, True]:
            validation_2d_histogram(
                report=report,
                x_true=x_true[:, 0],
                y_true=x_true[:, 2],
                range_x=[3.0, 153.0],
                range_y=[0, 800],
                weights_true=weights * y_true if weights else y_true,
                weights_pred=weights * y_pred if weights else y_pred,
                xlabel="Momentum [GeV/$c$]",
                ylabel="$\mathtt{nTracks}$",
                label_true="Full simulation" if from_fullsim else "Calibration samples",
                label_pred="ANN-based model",
                log_scale=log_scale,
                save_figure=save_images,
                export_fname=f"{images_dirname}/nTracks_vs_p-isMuon-hist",
            )

        identification_efficiency(
            report=report,
            x_true=x_true[:, 0],
            id_true=y_true,
            id_pred=y_pred,
            data_bin=x_true[:, 2],
            range_true=[3.0, 153.0],
            boundaries_bin=[0, 100, 250, 350, 800],
            weights_true=weights,
            weights_pred=weights,
            xlabel="Momentum [GeV/$c$]",
            label_model="isMuon",
            label_true="Full simulation" if from_fullsim else "Calibration samples",
            label_pred="ANN-based model",
            symbol_bin="$\mathtt{nTracks}$",
            save_figure=save_images,
            export_fname=f"{images_dirname}/nTracks_vs_p-isMuon-eff",
        )

        for log_scale in [False, True]:
            validation_2d_histogram(
                report=report,
                x_true=x_true[:, 1],
                y_true=x_true[:, 2],
                range_x=[1.5, 5.5],
                range_y=[0, 800],
                weights_true=weights * y_true if weights else y_true,
                weights_pred=weights * y_pred if weights else y_pred,
                xlabel="Pseudorapidity",
                ylabel="$\mathtt{nTracks}$",
                label_true="Full simulation" if from_fullsim else "Calibration samples",
                label_pred="ANN-based model",
                log_scale=log_scale,
                save_figure=save_images,
                export_fname=f"{images_dirname}/nTracks_vs_eta-isMuon-hist",
            )

        identification_efficiency(
            report=report,
            x_true=x_true[:, 1],
            id_true=y_true.flatten(),
            id_pred=y_pred.flatten(),
            data_bin=x_true[:, 2],
            range_true=[1.5, 5.5],
            boundaries_bin=[0, 100, 250, 350, 800],
            weights_true=weights,
            weights_pred=weights,
            xlabel="Pseudorapidity",
            label_model="isMuon",
            label_true="Full simulation" if from_fullsim else "Calibration samples",
            label_pred="ANN-based model",
            symbol_bin="$\mathtt{nTracks}$",
            save_figure=save_images,
            export_fname=f"{images_dirname}/nTracks_vs_p-isMuon-eff",
        )

        report.add_markdown("---")
