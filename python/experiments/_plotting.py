"""Figures for the five simulation scripts.

Sizes follow the MATLAB scripts (centimeters). Colors are the MATLAB default
order. Both covariance types of the noise figure share one color per method;
the original script lets the color-order index run on, which makes the dashed
curves a different color from the solid ones.
"""

from __future__ import annotations

import inspect

import numpy as np

MATLAB_COLORS = np.array(
    [
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ]
)

# Explicit colors from sim_accuracy_vs_noise_level.m, then the color the
# first covariance panel assigns once those explicit colors increment the order.
NOISE_COLOR = {
    1: MATLAB_COLORS[0],
    2: MATLAB_COLORS[0],
    3: MATLAB_COLORS[1],
    4: MATLAB_COLORS[4],
    5: MATLAB_COLORS[2],
    6: MATLAB_COLORS[3],
    7: MATLAB_COLORS[4],
    8: MATLAB_COLORS[5],
}
NOISE_MARKER = {1: "o", 2: "*", 3: "p", 4: "h", 5: "s", 6: "d", 7: "^", 8: "v"}


def _figure(width_cm: float, height_cm: float):
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), layout="constrained")


def _style(ax) -> None:
    ax.grid(True)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(labelsize=8)


def _boxplot(ax, values: np.ndarray, labels: list[str]):
    parameters = inspect.signature(ax.boxplot).parameters
    keyword = "tick_labels" if "tick_labels" in parameters else "labels"
    return ax.boxplot(values, **{keyword: labels})


def noise_figure(means_pos, means_rot, method_ids, legend, cov_names) -> object:
    """Mean position and orientation error against the noise setting."""
    figure = _figure(17.6, 8.8)
    axes = figure.subplots(1, 2)
    styles = ("-", "--")
    x = np.arange(1, means_pos.shape[0] + 1)
    for axis, values, title, ylabel in (
        (axes[0], means_pos, "Position error", r"$\varepsilon_{\mathbf{t}}$ (mm)"),
        (axes[1], means_rot, "Orientation error", r"$\varepsilon_{\mathbf{R}}$ (deg)"),
    ):
        _style(axis)
        for i_cov, cov in enumerate(cov_names):
            for i_method, method_id in enumerate(method_ids):
                axis.plot(
                    x,
                    values[:, i_cov, i_method],
                    styles[i_cov],
                    color=NOISE_COLOR[method_id],
                    marker=NOISE_MARKER[method_id],
                    markerfacecolor="white",
                    linewidth=1.25,
                    label=f"{legend[i_method]}, {cov}",
                )
        axis.set_xlim(1, means_pos.shape[0])
        axis.set_xticks(x)
        axis.set_xticklabels([f"#{i}" for i in x])
        axis.set_title(title)
        axis.set_xlabel("Noise setting")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=7, loc="best")
    return figure


def count_figure(measurements, time_mean, markers, legend) -> object:
    figure = _figure(8.8, 8.8)
    axis = figure.subplots()
    _style(axis)
    for index, label in enumerate(legend):
        axis.plot(
            measurements,
            time_mean[:, index],
            "-",
            color=MATLAB_COLORS[index % len(MATLAB_COLORS)],
            marker=markers[index],
            markerfacecolor="white",
            linewidth=1.25,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xlim(measurements[0], measurements[-1])
    axis.set_xticks(measurements)
    axis.set_xlabel("No. of measurements")
    axis.set_ylabel("Runtime (s)")
    axis.legend(fontsize=8, loc="best")
    return figure


def deviation_figures(rates, e_pos, e_rot, labels, sigmat, sigma_rd):
    bar = _figure(12, 8)
    axis = bar.subplots()
    _style(axis)
    x = np.arange(1, len(rates) + 1)
    axis.bar(x, rates, width=0.6, color=(0.3, 0.6, 0.9), edgecolor="black")
    for index, rate in enumerate(rates):
        axis.text(index + 1, rate + 2, f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)
    axis.set_xlim(0.5, len(rates) + 0.5)
    axis.set_ylim(0, 110)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15)
    axis.set_xlabel(r"Initial deviation $[\Delta\theta, \Delta t]$")
    axis.set_ylabel("Convergence rate (%)")
    axis.set_title(
        rf"Convergence rate ($\sigma_t={sigmat:.1f}$ mm, $\sigma_R={sigma_rd:.1f}^\circ$)"
    )

    boxes = _figure(14, 10)
    left, right = boxes.subplots(1, 2)
    for axis, values, ylabel, title in (
        (left, e_pos, r"$\varepsilon_{\mathbf{t}}$ (mm)", "Position error"),
        (right, e_rot, r"$\varepsilon_{\mathbf{R}}$ (deg)", "Orientation error"),
    ):
        _style(axis)
        _boxplot(axis, values, labels)
        axis.tick_params(axis="x", labelrotation=15)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Initial deviation")
        axis.set_title(title)

    histograms = _figure(18, 10)
    grid = histograms.subplots(2, e_pos.shape[1])
    for index in range(e_pos.shape[1]):
        _style(grid[0, index])
        grid[0, index].hist(e_pos[:, index], bins="auto", color=(0.3, 0.6, 0.9), edgecolor="black")
        grid[0, index].set_title(labels[index])
        grid[0, index].set_xlabel(r"$\varepsilon_{\mathbf{t}}$ (mm)")
        _style(grid[1, index])
        grid[1, index].hist(e_rot[:, index], bins="auto", color=(0.9, 0.5, 0.3), edgecolor="black")
        grid[1, index].set_xlabel(r"$\varepsilon_{\mathbf{R}}$ (deg)")
    grid[0, 0].set_ylabel("Frequency")
    grid[1, 0].set_ylabel("Frequency")
    return bar, boxes, histograms


def vfe_figure(e_rot, e_pos) -> object:
    figure = _figure(8.8, 6.6)
    left, right = figure.subplots(1, 2)
    labels = ["VFE", "GT", "EQ"]
    for axis, values, ylabel in (
        (left, e_rot, r"$\varepsilon_{\mathbf{R}}$ (deg)"),
        (right, e_pos, r"$\varepsilon_{\mathbf{t}}$ (mm)"),
    ):
        _style(axis)
        artists = _boxplot(axis, values, labels)
        for line in artists["medians"]:
            line.set_linewidth(2.0)
        for key in ("boxes", "whiskers"):
            for line in artists[key]:
                line.set_linewidth(1.2)
        axis.set_ylabel(ylabel)
    return figure
