from utils_plot import (
    binned_validation_histogram,
    correlation_histogram,
    learn_rate_scheduling,
    learning_curves,
    metric_curves,
    selection_efficiency,
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
    history,
    metrics,
    num_epochs,
    loss_name=None,
    is_from_validation_set=True,
    save_images=False,
    images_dirname="./images",
) -> None:
    report.add_markdown('<h2 align="center">Training plots</h2>')

    start_epoch = int(num_epochs / 20)

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
    x_true,
    y_true,
    y_pred,
    y_vars,
    weights=None,
    is_from_fullsim=True,
    save_images=False,
    images_dirname="./images",
) -> None:
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
            label_ref="Full simulation" if is_from_fullsim else "Calibration samples",
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
            label_ref="Full simulation" if is_from_fullsim else "Calibration samples",
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
            label_ref="Full simulation" if is_from_fullsim else "Calibration samples",
            label_gen="GAN-based model",
            save_figure=save_images,
            export_fname=f"{images_dirname}/{y_var}_vs_nTracks-eff",
        )

        report.add_markdown("---")
