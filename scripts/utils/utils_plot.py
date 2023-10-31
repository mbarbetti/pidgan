import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

my_cmap = copy.copy(mpl.cm.get_cmap("magma"))
my_cmap.set_bad((0, 0, 0))


def learning_curves(
    report,
    history,
    start_epoch=0,
    keys=["loss"],
    colors=None,
    labels=None,
    legend_loc="upper right",
    save_figure=False,
    scale_curves=True,
    export_fname="./images/learn-curves",
) -> None:
    if colors is None:
        colors = [None for _ in range(len(keys))]
    else:
        assert len(colors) == len(keys)

    if labels is None:
        labels = [None for _ in range(len(keys))]
    else:
        assert len(labels) == len(keys)

    if scale_curves:
        if "g_loss" in keys:
            main_key = "g_loss"
        else:
            main_key = keys[0]
        exp_dict = dict()
        for k in keys:
            exp_dict[k] = np.round(
                np.mean(
                    np.log(np.abs(np.array(history[k])[start_epoch:])) / np.log(10.0),
                    axis=-1,
                )
            )
        scales = list()
        for i in range(len(keys)):
            exp = int(exp_dict[main_key] - exp_dict[keys[i]])
            scales.append(10**exp)
            if labels[i] is not None:
                labels[i] += " [x $10^{" + f"{exp}" + "}$]"
    else:
        scales = [1.0 for _ in keys]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.title("Learning curves", fontsize=14)
    plt.xlabel("Training epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        num_epochs = np.arange(len(history[key]))[start_epoch:]
        loss = np.array(history[key])[start_epoch:] * scales[i]
        plt.plot(num_epochs, loss, lw=1.5, color=color, label=label)
    plt.legend(loc=legend_loc, fontsize=10)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=45%")
    plt.close()


def learn_rate_scheduling(
    report,
    history,
    start_epoch=0,
    keys=["lr"],
    colors=None,
    labels=None,
    legend_loc="upper right",
    save_figure=False,
    export_fname="./images/lr-sched",
) -> None:
    if colors is None:
        colors = [None for _ in range(len(keys))]
    else:
        assert len(colors) == len(keys)

    if labels is None:
        labels = [None for _ in range(len(keys))]
    else:
        assert len(labels) == len(keys)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.title("Learning rate scheduling", fontsize=14)
    plt.xlabel("Training epochs", fontsize=12)
    plt.ylabel("Learning rate", fontsize=12)
    for key, label, color in zip(keys, labels, colors):
        num_epochs = np.arange(len(history[key]))[start_epoch:]
        lr = np.array(history[key])[start_epoch:]
        plt.plot(num_epochs, lr, lw=1.5, color=color, label=label)
    plt.legend(loc=legend_loc, fontsize=10)
    plt.yscale("log")
    if save_figure:
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=45%")
    plt.close()


def metric_curves(
    report,
    history,
    start_epoch=0,
    key="metric",
    ylabel="Metric",
    title="Metric curves",
    validation_set=False,
    colors=None,
    labels=None,
    legend_loc="upper right",
    yscale="linear",
    save_figure=False,
    export_fname="./images/metric-curves",
) -> None:
    keys = [key]
    if validation_set:
        keys += [f"val_{key}"]

    if colors is None:
        colors = [None for _ in range(len(keys))]
    else:
        colors = colors[: len(keys)]

    if labels is None:
        labels = [None for _ in range(len(keys))]
    else:
        labels = labels[: len(keys)]

    zorder = 1
    plt.figure(figsize=(8, 5), dpi=300)
    plt.title(title, fontsize=14)
    plt.xlabel("Training epochs", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    for key, label, color in zip(keys, labels, colors):
        num_epochs = np.arange(len(history[key]))[start_epoch:]
        metric = np.array(history[key])[start_epoch:]
        plt.plot(num_epochs, metric, lw=1.5, color=color, label=label, zorder=zorder)
        zorder -= 1
    plt.legend(loc=legend_loc, fontsize=10)
    plt.yscale(yscale)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=45%")
    plt.close()


def validation_histogram(
    report,
    data_ref,
    data_gen,
    weights_ref=None,
    weights_gen=None,
    xlabel=None,
    density=False,
    label_ref="Data sample",
    label_gen="GAN model",
    log_scale=False,
    legend_loc=None,
    save_figure=False,
    export_fname="./images/val-hist",
) -> None:
    min_ = data_ref.mean() - 4.0 * data_ref.std()
    max_ = data_ref.mean() + 4.0 * data_ref.std()
    bins = np.linspace(min_, max_, 101)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Candidates", fontsize=12)
    plt.hist(
        data_ref,
        bins=bins,
        density=density,
        weights=weights_ref,
        color="#3288bd",
        label=label_ref,
    )
    plt.hist(
        data_gen,
        bins=bins,
        density=density,
        weights=weights_gen,
        histtype="step",
        color="#fc8d59",
        lw=2,
        label=label_gen,
    )
    if log_scale:
        plt.yscale("log")
        export_fname += "-log"
    plt.legend(loc=legend_loc, fontsize=10)
    if save_figure:
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=45%")
    plt.close()


def correlation_histogram(
    report,
    data_corr,
    data_ref,
    data_gen,
    range_corr,
    weights_ref=None,
    weights_gen=None,
    xlabel=None,
    ylabel=None,
    label_ref="Data sample",
    label_gen="GAN model",
    log_scale=False,
    save_figure=False,
    export_fname="./images/corr-hist",
) -> None:
    x_min, x_max = range_corr
    x_bins = np.linspace(x_min, x_max, 76)

    y_min = data_ref.mean() - 4.0 * data_ref.std()
    y_max = data_ref.mean() + 4.0 * data_ref.std()
    y_bins = np.linspace(y_min, y_max, 76)

    Y = [data_ref, data_gen]
    W = [weights_ref, weights_gen]
    bins = [x_bins, y_bins]
    titles = [label_ref, label_gen]

    vmin = 1.0 if log_scale else 0.0
    vmax = 0.0
    for data_y, weights in zip(Y, W):
        h, _, _ = np.histogram2d(data_corr, data_y, bins=bins, weights=weights)
        vmax = max(h.max(), vmax)

    plt.figure(figsize=(18, 5), dpi=300)
    for i, (data_y, weights, title) in enumerate(zip(Y, W, titles)):
        plt.subplot(1, 2, i + 1)
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.hist2d(
            data_corr,
            data_y,
            norm=mpl.colors.LogNorm(vmin=vmin) if log_scale else None,
            weights=weights,
            bins=bins,
            cmap=my_cmap,
        )
        plt.clim(vmin=vmin, vmax=vmax)
        plt.colorbar()
    if save_figure:
        if log_scale:
            export_fname += "-log"
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=95%")
    plt.close()


def binned_validation_histogram(
    report,
    data_bin,
    data_ref,
    data_gen,
    boundaries_bin,
    weights_ref=None,
    weights_gen=None,
    xlabel=None,
    density=False,
    label_ref="Data sample",
    label_gen="GAN model",
    symbol_bin=None,
    unit_bin=None,
    log_scale=False,
    save_figure=False,
    export_fname="./images/bin-val-hist",
) -> None:
    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10), dpi=300)

    idx = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel(xlabel, fontsize=12)
            ax[i, j].set_ylabel("Candidates", fontsize=12)

            query = (data_bin >= boundaries_bin[idx]) & (
                data_bin < boundaries_bin[idx + 1]
            )
            min_ = data_ref[query].mean() - 4.0 * data_ref[query].std()
            max_ = data_ref[query].mean() + 4.0 * data_ref[query].std()
            bins = np.linspace(min_, max_, 101)

            ax[i, j].hist(
                data_ref[query],
                bins=bins,
                density=density,
                weights=weights_ref,
                color="#3288bd",
                label=label_ref,
            )
            ax[i, j].hist(
                data_gen[query],
                bins=bins,
                density=density,
                weights=weights_gen,
                histtype="step",
                color="#fc8d59",
                lw=2,
                label=label_gen,
            )

            if symbol_bin is not None:
                text = f"{symbol_bin}"
            else:
                text = "Condition"
            text += f" $\in ({boundaries_bin[idx]:.1f}, {boundaries_bin[idx + 1]:.1f})$"
            if unit_bin is not None:
                text += f" {unit_bin}"
            ax[i, j].annotate(
                text,
                fontsize=10,
                ha="right",
                va="top",
                xy=(0.95, 0.95),
                xycoords="axes fraction",
            )

            if log_scale:
                ax[i, j].set_yscale("log")
            ax[i, j].legend(loc="upper left", fontsize=10)

            idx += 1

    if save_figure:
        if log_scale:
            export_fname += "-log"
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=45%")
    plt.close()


def selection_efficiency(
    report,
    data_bin,
    data_ref,
    data_gen,
    range_bin,
    weights_ref=None,
    weights_gen=None,
    title=None,
    xlabel=None,
    label_ref="Data sample",
    label_gen="GAN model",
    save_figure=False,
    export_fname="./images/sel-eff",
) -> None:
    min_, max_ = range_bin
    bins = np.linspace(min_, max_, 26)

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 5), dpi=300)

    for i, (pctl, ylabel) in enumerate(zip([25, 50, 75], ["Loose", "Mild", "Tight"])):
        ax[i].set_title(f"{title} > $Q_{i + 1}$", fontsize=14)
        ax[i].set_xlabel(xlabel, fontsize=12)
        ax[i].set_ylabel(f"{ylabel} selection efficiency", fontsize=12)

        selection = np.percentile(data_ref, pctl)
        query_ref = data_ref > selection
        query_gen = data_gen > selection

        h_tot, _ = np.histogram(data_bin, bins=bins, weights=weights_ref)
        h_ref, _ = np.histogram(
            data_bin[query_ref],
            bins=bins,
            weights=weights_ref[query_ref] if weights_ref else None,
        )
        h_gen, _ = np.histogram(
            data_bin[query_gen],
            bins=bins,
            weights=weights_gen[query_gen] if weights_gen else None,
        )

        h_tot_err = np.where(h_tot > 0.0, np.sqrt(h_tot) / (h_tot + 1e-12), 0.0)
        h_ref_err = np.where(h_ref > 0.0, np.sqrt(h_ref) / (h_ref + 1e-12), 0.0)
        h_gen_err = np.where(h_gen > 0.0, np.sqrt(h_gen) / (h_gen + 1e-12), 0.0)

        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        eff_ref = np.where(h_tot > 0.0, h_ref / (h_tot + 1e-12), 0.0)
        eff_gen = np.where(h_tot > 0.0, h_gen / (h_tot + 1e-12), 0.0)
        eff_ref_err = eff_ref * np.sqrt(h_tot_err**2 + h_ref_err**2)
        eff_gen_err = eff_gen * np.sqrt(h_tot_err**2 + h_gen_err**2)

        ax[i].errorbar(
            bin_centers,
            eff_ref,
            yerr=eff_ref_err,
            marker="^",
            markersize=3,
            capsize=3,
            elinewidth=2,
            color="#3288bd",
            label=label_ref,
            zorder=0,
        )
        ax[i].errorbar(
            bin_centers,
            eff_gen,
            yerr=eff_gen_err,
            marker="v",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color="#fc8d59",
            label=label_gen,
            zorder=1,
        )
        ax[i].legend(loc=None, fontsize=10)

    if save_figure:
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=95%")
    plt.close()


def validation_2d_histogram(
    report,
    x_true,
    y_true,
    range_x,
    range_y,
    weights_true=None,
    weights_pred=None,
    xlabel=None,
    ylabel=None,
    label_true="Data sample",
    label_pred="ANN model",
    log_scale=False,
    save_figure=False,
    export_fname="./images/val-2d-hist",
) -> None:
    x_min, x_max = range_x
    x_bins = np.linspace(x_min, x_max, 76)

    y_min, y_max = range_y
    y_bins = np.linspace(y_min, y_max, 76)

    W = [weights_true, weights_pred]
    bins = [x_bins, y_bins]
    titles = [label_true, label_pred]

    vmin = 1.0 if log_scale else 0.0
    vmax = 0.0
    for weights in W:
        h, _, _ = np.histogram2d(x_true, y_true, bins=bins, weights=weights)
        vmax = max(h.max(), vmax)

    plt.figure(figsize=(18, 5), dpi=300)
    for i, (weights, title) in enumerate(zip(W, titles)):
        plt.subplot(1, 2, i + 1)
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.hist2d(
            x_true,
            y_true,
            norm=mpl.colors.LogNorm(vmin=vmin) if log_scale else None,
            weights=weights,
            bins=bins,
            cmap=my_cmap,
        )
        plt.clim(vmin=vmin, vmax=vmax)
        plt.colorbar()
    if save_figure:
        if log_scale:
            export_fname += "-log"
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=95%")
    plt.close()


def identification_efficiency(
    report,
    x_true,
    id_true,
    id_pred,
    data_bin,
    range_true,
    boundaries_bin,
    weights_true=None,
    weights_pred=None,
    xlabel=None,
    label_model="isMuon",
    label_true="Data sample",
    label_pred="ANN model",
    symbol_bin=None,
    unit_bin=None,
    save_figure=False,
    export_fname="./images/id-eff",
) -> None:
    min_, max_ = range_true
    bins = np.linspace(min_, max_, 26)

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 20), dpi=300)
    fig.tight_layout(pad=5)

    for i in range(4):
        ax[i, 0].set_xlabel(xlabel, fontsize=12)
        ax[i, 0].set_ylabel("Candidates", fontsize=12)

        query = (data_bin >= boundaries_bin[i]) & (data_bin < boundaries_bin[i + 1])

        h_tot, _ = np.histogram(x_true, bins=bins, weights=weights_true)
        h_true, _ = np.histogram(
            x_true[query],
            bins=bins,
            weights=weights_true[query] * id_true[query]
            if weights_true
            else id_true[query],
        )
        h_pred, _ = np.histogram(
            x_true[query],
            bins=bins,
            weights=weights_pred[query] * id_pred[query]
            if weights_pred
            else id_pred[query],
        )

        h_tot_err = np.where(h_tot > 0.0, np.sqrt(h_tot) / (h_tot + 1e-12), 0.0)
        h_true_err = np.where(h_true > 0.0, np.sqrt(h_true) / (h_true + 1e-12), 0.0)
        h_pred_err = np.where(h_pred > 0.0, np.sqrt(h_pred) / (h_pred + 1e-12), 0.0)

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        ax[i, 0].hist(
            x_true,
            bins=bins,
            weights=weights_true,
            histtype="step",
            color="#d01c8b",
            lw=2,
            label=label_true,
        )
        ax[i, 0].hist(
            x_true[query],
            bins=bins,
            weights=weights_pred[query] * id_pred[query]
            if weights_pred
            else id_pred[query],
            histtype="step",
            color="#4dac26",
            lw=1.5,
            label=label_pred,
        )
        ax[i, 0].errorbar(
            bin_centers,
            h_true,
            yerr=h_true_err * h_true,
            fmt=".",
            color="black",
            barsabove=True,
            capsize=2,
            label=f"{label_model} passed",
        )
        ax[i, 0].set_yscale("log")
        ax[i, 0].set_ylim(bottom=1, top=5 * h_tot.max())

        if symbol_bin is not None:
            text = f"{symbol_bin}"
        else:
            text = "Condition"
        text += f" $\in ({boundaries_bin[i]:.1f}, {boundaries_bin[i + 1]:.1f})$"
        if unit_bin is not None:
            text += f" {unit_bin}"

        ax[i, 0].annotate(
            text,
            fontsize=10,
            ha="left",
            va="top",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
        )
        ax[i, 0].legend(loc="upper right", fontsize=10)

        ax[i, 1].set_xlabel(xlabel, fontsize=12)
        ax[i, 1].set_ylabel(f"{label_model} efficiency", fontsize=12)

        eff_true = np.where(h_tot > 0.0, h_true / (h_tot + 1e-12), 0.0)
        eff_pred = np.where(h_tot > 0.0, h_pred / (h_tot + 1e-12), 0.0)
        eff_true_err = eff_true * np.sqrt(h_tot_err**2 + h_true_err**2)
        eff_pred_err = eff_pred * np.sqrt(h_tot_err**2 + h_pred_err**2)

        ax[i, 1].errorbar(
            bin_centers,
            eff_true,
            yerr=eff_true_err,
            marker="^",
            markersize=3,
            capsize=3,
            elinewidth=2,
            color="#3288bd",
            label=label_true,
            zorder=0,
        )
        ax[i, 1].errorbar(
            bin_centers,
            eff_pred,
            yerr=eff_pred_err,
            marker="v",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color="#fc8d59",
            label=label_pred,
            zorder=1,
        )
        y_max = max(
            (eff_true + 0.5 * eff_true_err).max(), (eff_pred + 0.5 * eff_pred_err).max()
        )
        ax[i, 1].set_ylim(bottom=-0.1 * y_max, top=1.3 * y_max)

        ax[i, 1].annotate(
            text,
            fontsize=10,
            ha="right",
            va="top",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
        )
        ax[i, 1].legend(loc="upper left", fontsize=10)

    if save_figure:
        plt.savefig(f"{export_fname}.png")
        print(f"[INFO] Figure correctly exported to '{export_fname}.png'")
    report.add_figure(options="width=75%")
    plt.close()
