import numpy as np

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


def prepare_training_plots(
    report,
    model,
    history,
    metrics=None,
    num_epochs=None,
    loss_name=None,
    is_from_validation_set=True,
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
            keys=["g_loss", "d_loss"],
            colors=["#3288bd", "#fc8d59"],
            labels=["generator", "discriminator"],
            legend_loc=None,
            save_figure=save_images,
            scale_curves=False,
            export_fname=f"{images_dirname}/learn-curves",
        )

        learn_rate_scheduling(
            report=report,
            history=history,
            start_epoch=0,
            keys=["g_lr", "d_lr"],
            colors=["#3288bd", "#fc8d59"],
            labels=["generator", "discriminator"],
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
            validation_set=is_from_validation_set,
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
            validation_set=is_from_validation_set,
            colors=["#d01c8b", "#4dac26"],
            labels=["training set", "validation set"],
            legend_loc=None,
            yscale="linear",
            save_figure=save_images,
            export_fname=f"{images_dirname}/disc-loss",
        )

    else:
        metric_curves(
            report=report,
            history=history,
            start_epoch=start_epoch,
            key="loss",
            ylabel="Binary cross-entropy",
            title="Learning curves",
            validation_set=is_from_validation_set,
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
                validation_set=is_from_validation_set,
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
    is_from_fullsim=True,
    save_images=False,
    images_dirname="./images",
) -> None:
    if model != "isMuon":
        if model == "Rich":
            idx_p = y_vars.index("RichDLLp")
            idx_k = y_vars.index("RichDLLk")
            new_y_true = y_true[:, idx_p] - y_true[:, idx_k]
            y_true = np.c_[y_true, new_y_true]
            new_y_pred = y_pred[:, idx_p] - y_pred[:, idx_k]
            y_pred = np.c_[y_pred, new_y_pred]
            y_vars += ["RichDLLpk"]
        elif model == "Muon":
            idx_mu = y_vars.index("MuonMuLL")
            idx_bg = y_vars.index("MuonBgLL")
            new_y_true = y_true[:, idx_mu] - y_true[:, idx_bg]
            y_true = np.c_[y_true, new_y_true]
            new_y_pred = y_pred[:, idx_mu] - y_pred[:, idx_bg]
            y_pred = np.c_[y_pred, new_y_pred]
            y_vars += ["MuonLL"]
        elif "GlobalPID" in model:
            idx_p = y_vars.index("PIDp")
            idx_k = y_vars.index("PIDK")
            new_y_true = y_true[:, idx_p] - y_true[:, idx_k]
            y_true = np.c_[y_true, new_y_true]
            new_y_pred = y_pred[:, idx_p] - y_pred[:, idx_k]
            y_pred = np.c_[y_pred, new_y_pred]
            y_vars += ["PIDpk"]

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
                    if is_from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}-hist",
                )

            for log_scale in [False, True]:
                correlation_histogram(
                    report=report,
                    data_corr=x_true[:, 0] / 1e3,
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    range_corr=[0, 100],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel="Momentum [GeV/$c$]",
                    ylabel=f"{y_var}",
                    label_ref="Full simulation"
                    if is_from_fullsim
                    else "Calibration samples",
                    label_gen="GAN-based model",
                    log_scale=log_scale,
                    save_figure=save_images,
                    export_fname=f"{images_dirname}/{y_var}_vs_p-corr_hist",
                )

            for log_scale in [False, True]:
                binned_validation_histogram(
                    report=report,
                    data_bin=x_true[:, 0] / 1e3,
                    data_ref=y_true[:, i],
                    data_gen=y_pred[:, i],
                    boundaries_bin=[0, 5, 10, 25, 100],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if is_from_fullsim
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
                data_bin=x_true[:, 0] / 1e3,
                data_ref=y_true[:, i],
                data_gen=y_pred[:, i],
                range_bin=[0, 100],
                weights_ref=weights,
                weights_gen=weights,
                title=f"{y_var}",
                xlabel="Momentum [GeV/$c$]",
                label_ref="Full simulation"
                if is_from_fullsim
                else "Calibration samples",
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
                    range_corr=[2, 5],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel="Pseudorapidity",
                    ylabel=f"{y_var}",
                    label_ref="Full simulation"
                    if is_from_fullsim
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
                    boundaries_bin=[1.8, 2.7, 3.5, 4.2, 5.5],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if is_from_fullsim
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
                range_bin=[1.8, 5.5],
                weights_ref=weights,
                weights_gen=weights,
                title=f"{y_var}",
                xlabel="Pseudorapidity",
                label_ref="Full simulation"
                if is_from_fullsim
                else "Calibration samples",
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
                    range_corr=[0, 500],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel="$\mathtt{nTracks}$",
                    ylabel=f"{y_var}",
                    label_ref="Full simulation"
                    if is_from_fullsim
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
                    boundaries_bin=[0, 50, 150, 300, 500],
                    weights_ref=weights,
                    weights_gen=weights,
                    xlabel=f"{y_var}",
                    label_ref="Full simulation"
                    if is_from_fullsim
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
                range_bin=[0, 500],
                weights_ref=weights,
                weights_gen=weights,
                title=f"{y_var}",
                xlabel="$\mathtt{nTracks}$",
                label_ref="Full simulation"
                if is_from_fullsim
                else "Calibration samples",
                label_gen="GAN-based model",
                save_figure=save_images,
                export_fname=f"{images_dirname}/{y_var}_vs_nTracks-eff",
            )

            report.add_markdown("---")

    else:
        report.add_markdown(f'<h2 align="center">Validation plots</h2>')

        for log_scale in [False, True]:
            validation_2d_histogram(
                report=report,
                x_true=x_true[:, 0] / 1e3,
                y_true=x_true[:, 1],
                range_x=[0, 100],
                range_y=[2, 5],
                weights_true=weights * y_true if weights else y_true,
                weights_pred=weights * y_pred if weights else y_pred,
                xlabel="Momentum [GeV/$c$]",
                ylabel="Pseudorapidity",
                label_true="Full simulation"
                if is_from_fullsim
                else "Calibration samples",
                label_pred="ANN-based model",
                log_scale=log_scale,
                save_figure=save_images,
                export_fname=f"{images_dirname}/eta_vs_p-isMuon-hist",
            )

        identification_efficiency(
            report=report,
            x_true=x_true[:, 0] / 1e3,
            id_true=y_true,
            id_pred=y_pred,
            data_bin=x_true[:, 1],
            range_true=[0, 100],
            boundaries_bin=[1.8, 2.7, 3.5, 4.2, 5.5],
            weights_true=weights,
            weights_pred=weights,
            xlabel="Momentum [GeV/$c$]",
            label_model="isMuon",
            label_true="Full simulation" if is_from_fullsim else "Calibration samples",
            label_pred="ANN-based model",
            symbol_bin="$\eta$",
            save_figure=save_images,
            export_fname=f"{images_dirname}/eta_vs_p-isMuon-eff",
        )

        for log_scale in [False, True]:
            validation_2d_histogram(
                report=report,
                x_true=x_true[:, 0] / 1e3,
                y_true=x_true[:, 2],
                range_x=[0, 100],
                range_y=[0, 500],
                weights_true=weights * y_true if weights else y_true,
                weights_pred=weights * y_pred if weights else y_pred,
                xlabel="Momentum [GeV/$c$]",
                ylabel="$\mathtt{nTracks}$",
                label_true="Full simulation"
                if is_from_fullsim
                else "Calibration samples",
                label_pred="ANN-based model",
                log_scale=log_scale,
                save_figure=save_images,
                export_fname=f"{images_dirname}/nTracks_vs_p-isMuon-hist",
            )

        identification_efficiency(
            report=report,
            x_true=x_true[:, 0] / 1e3,
            id_true=y_true,
            id_pred=y_pred,
            data_bin=x_true[:, 2],
            range_true=[0, 100],
            boundaries_bin=[0, 50, 150, 300, 500],
            weights_true=weights,
            weights_pred=weights,
            xlabel="Momentum [GeV/$c$]",
            label_model="isMuon",
            label_true="Full simulation" if is_from_fullsim else "Calibration samples",
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
                range_x=[2, 5],
                range_y=[0, 500],
                weights_true=weights * y_true if weights else y_true,
                weights_pred=weights * y_pred if weights else y_pred,
                xlabel="Pseudorapidity",
                ylabel="$\mathtt{nTracks}$",
                label_true="Full simulation"
                if is_from_fullsim
                else "Calibration samples",
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
            range_true=[2, 5],
            boundaries_bin=[0, 50, 150, 300, 500],
            weights_true=weights,
            weights_pred=weights,
            xlabel="Pseudorapidity",
            label_model="isMuon",
            label_true="Full simulation" if is_from_fullsim else "Calibration samples",
            label_pred="ANN-based model",
            symbol_bin="$\mathtt{nTracks}$",
            save_figure=save_images,
            export_fname=f"{images_dirname}/nTracks_vs_p-isMuon-eff",
        )

        report.add_markdown("---")
