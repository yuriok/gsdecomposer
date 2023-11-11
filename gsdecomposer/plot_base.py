__all__ = ["summarize", "plot_gsds", "plot_components", "plot_proportions"]

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import font_manager
import scienceplots

from QGrain.utils import get_image_by_proportions

for font in os.listdir("./fonts"):
    font_manager.fontManager.addfont(os.path.join("./fonts", font))

plt.style.use(["science", "no-latex"])
# plt.rcParams["font.family"] = "Source Han Sans CN"
plt.rcParams["savefig.transparent"] = False
plt.rcParams["savefig.dpi"] = 300.0
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["mathtext.fontset"] = "stix"


def summarize(components: np.ndarray, q=0.01):
    mean = np.mean(components, axis=0)
    upper = np.quantile(components, q=1 - q, axis=0)
    lower = np.quantile(components, q=q, axis=0)
    return mean, lower, upper


def summarize_components(proportions: np.ndarray, components: np.ndarray, q=0.01):
    mean = np.zeros((components.shape[1], components.shape[2]))
    upper = np.zeros((components.shape[1], components.shape[2]))
    lower = np.zeros((components.shape[1], components.shape[2]))
    for i in range(components.shape[1]):
        key = proportions[:, 0, i] > 1e-3
        mean[i] = np.mean(components[:, i, :][key], axis=0)
        upper[i] = np.quantile(components[:, i, :][key], q=1 - q, axis=0)
        lower[i] = np.quantile(components[:, i, :][key], q=q, axis=0)
    return mean, lower, upper


def plot_gsds(
        axes: plt.Axes,
        classes: np.ndarray,
        distributions: np.ndarray,
        xlabel=r"Grain size (μm)",
        ylabel=r"Frequency (%)",
        xlim=(0.06, 2000),
        ylim=(0.0, 10.5),
        xticks=(0.1, 1, 10, 100, 1000),
        yticks=(2, 6, 10),
        title=None):
    colors = ["#000000", "#808080", "#e0e0e0"]
    mean, lower, upper = summarize(distributions, q=0.01)
    axes.fill_between(classes, lower * 100, upper * 100,
                      color=colors[2], lw=0.02, zorder=-10)
    mean, lower, upper = summarize(distributions, q=0.05)
    axes.fill_between(classes, lower * 100, upper * 100,
                      color=colors[1], lw=0.02, zorder=-10)
    axes.plot(classes, mean * 100, color=colors[0], zorder=-10)
    axes.set_xscale("log")
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    axes.set_xticks(xticks, [str(tick) for tick in xticks])
    axes.set_yticks(yticks, [str(tick) for tick in yticks])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)


def plot_components(
        axes: plt.Axes,
        classes: np.ndarray,
        proportions: np.ndarray,
        components: np.ndarray,
        xlabel=r"Grain size (μm)",
        ylabel=r"Frequency (%)",
        xlim=(0.06, 2000),
        ylim=(0.0, 10.5),
        xticks=(0.1, 1, 10, 100, 1000),
        yticks=(2, 6, 10),
        title=None):
    n_components = components.shape[1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    light_colors = ["#A8D2F0", "#FFC999", "#AFE9AF", "#EFA9AA", "#CDB8E0"]
    mean, lower, upper = summarize_components(proportions, components, q=0.01)
    for i in range(n_components):
        axes.fill_between(
            classes, lower[i] * 100, upper[i] * 100,
            color=light_colors[i], lw=0.02, zorder=-10 + i)
        axes.plot(classes, mean[i] * 100, color=colors[i], zorder=-10 + i)
    axes.set_xscale("log")
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    axes.set_xticks(xticks, [str(tick) for tick in xticks])
    axes.set_yticks(yticks, [str(tick) for tick in yticks])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)


def plot_proportions(
        axes: plt.Axes,
        proportions: np.ndarray,
        xlabel="Sample index",
        ylabel="Proportion (%)",
        cmap="tab10",
        vmin=0,
        vmax=9,
        title=None):
    n_samples, _, _ = proportions.shape
    image = get_image_by_proportions(proportions[:, 0, :], resolution=100)
    axes.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=(0.0, n_samples, 100.0, 0.0))
    axes.set_ylim(0.0, 100.0)
    axes.set_yticks([20, 40, 60, 80], ["20", "40", "60", "80"])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
