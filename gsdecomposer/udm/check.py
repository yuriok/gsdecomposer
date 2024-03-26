import os
import pickle
import string

import matplotlib.lines
import numpy as np
import matplotlib.pyplot as plt
from QGrain.models import Dataset, UDMResult
from QGrain.distributions import GeneralWeibull

from gsdecomposer.plot_base import *

sections = ("GJP", "BGY", "YB19",
            "FS18", "YC", "LC",
            "TC", "WN19", "BL",
            "LX", "BSK", "CMG",
            "NLK", "Osh", "HKG",
            "BS", "HX", "WB1")

loess_sections = ("GJP", "BGY", "YC", "LC", "TC", "BL", "WN19", "YB19",
                  "LX", "FS18", "Osh", "BSK", "CMG", "NLK")
fluvial_sections = ("HKG", "BS")
lake_delta_sections = ("HX", "WB1")
selected_sections = ("GJP", "WN19", "HKG", "BS", "HX", "WB1")

cmap = plt.get_cmap("tab10")
for i, section in enumerate(sections + selected_sections):
    if section in loess_sections:
        facies = "loess"
    elif section in fluvial_sections:
        facies = "fluvial"
    elif section in lake_delta_sections:
        facies = "lake_delta"
    else:
        raise NotImplementedError(section)
    with open(f"./datasets/UDM/{facies}/{section}.udm", "rb") as f:
        udm_result: UDMResult = pickle.load(f)

    if i % 6 == 0:
        plt.figure(figsize=(6.6, 8.8))
    dataset: Dataset = udm_result.dataset
    classes = np.expand_dims(np.expand_dims(dataset.classes_phi, axis=0), axis=0).repeat(
        len(dataset), axis=0).repeat(udm_result.n_components, axis=1)
    interval = np.abs((dataset.classes_phi[0] - dataset.classes_phi[-1]) / (dataset.n_classes - 1))
    proportions_udm, components_udm, (m_udm, std_udm, s_udm, k_udm) = GeneralWeibull.interpret(
        udm_result.parameters[-1], classes, interval)
    if facies == "loess":
        xlim = (0.06, 600)
        xticks = (0.1, 1, 10, 100)
    elif facies == "fluvial":
        xlim = (0.06, 2000)
        xticks = (0.1, 1, 10, 100, 1000)
    else:
        xlim = (0.06, 5000)
        xticks = (0.1, 1, 10, 100, 5000)
    axes_1 = plt.subplot(6, 3, 1 + (i % 6) * 3)
    plot_gsds(axes_1, dataset.classes, dataset.distributions,
              xlim=xlim, xticks=xticks, title=section)
    axes_2 = plt.subplot(6, 3, 2 + (i % 6) * 3)
    plot_components(axes_2, dataset.classes, proportions_udm, components_udm,
                    xlim=xlim, xticks=xticks, title=section)

    axes_3 = plt.subplot(6, 3, 3 + (i % 6) * 3)
    plot_proportions(axes_3, proportions_udm, title=section)
    if i % 6 == 5:
        plt.tight_layout()
        for n, ax in enumerate(plt.gcf().axes):
            ax.text(-0.15, 1.06,
                    f"{string.ascii_uppercase[n]}",
                    transform=ax.transAxes,
                    size=10, weight="bold")
        if i//6 < 2:
            handles = [matplotlib.patches.Rectangle([0, 0], 1, 1, color=cmap(i)) for i in range(3)]
            labels = [f"C{i+1}" for i in range(3)]
            axes_2.legend(handles, labels,
                          loc="lower center", bbox_to_anchor=(0.5, -0.75),
                          ncols=3, prop={"size": 6}, columnspacing=20.0)
        else:
            handles = [matplotlib.patches.Rectangle([0, 0], 1, 1, color=cmap(i)) for i in range(5)]
            labels = [f"C{i + 1}" for i in range(5)]
            axes_2.legend(handles, labels,
                          loc="lower center", bbox_to_anchor=(0.5, -0.75),
                          ncols=5, prop={"size": 6}, columnspacing=10.0)
        os.makedirs("./figures/udm", exist_ok=True)
        plt.savefig(f"./figures/udm/results_{i//6+1}.svg")
        plt.savefig(f"./figures/udm/results_{i//6+1}.eps")
        plt.close()

plt.figure(figsize=(6.6, 8.8))
cmap = plt.get_cmap("tab10")
for i, section in enumerate(sections):
    if section in loess_sections:
        facies = "loess"
    elif section in fluvial_sections:
        facies = "fluvial"
    elif section in lake_delta_sections:
        facies = "lake_delta"
    else:
        raise NotImplementedError(section)
    with open(f"./datasets/UDM/{facies}/{section}.udm", "rb") as f:
        udm_result: UDMResult = pickle.load(f)
    dataset: Dataset = udm_result.dataset
    classes = np.expand_dims(np.expand_dims(dataset.classes_phi, axis=0), axis=0).repeat(
        len(dataset), axis=0).repeat(udm_result.n_components, axis=1)
    interval = np.abs((dataset.classes_phi[0] - dataset.classes_phi[-1]) / (dataset.n_classes - 1))
    proportions_udm, components_udm, (m_udm, std_udm, s_udm, k_udm) = GeneralWeibull.interpret(
        udm_result.parameters[-1], classes, interval)
    if facies == "loess":
        xlim = (0.06, 600)
        xticks = (0.1, 1, 10, 100)
    elif facies == "fluvial":
        xlim = (0.06, 2000)
        xticks = (0.1, 1, 10, 100, 1000)
    else:
        xlim = (0.06, 5000)
        xticks = (0.1, 1, 10, 100, 5000)
    axes = plt.subplot(6, 3, i + 1)
    plot_components(axes, dataset.classes, proportions_udm, components_udm,
                    xlim=xlim, xticks=xticks, title=section)
plt.tight_layout()
for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06,
            f"{string.ascii_uppercase[n]}",
            transform=ax.transAxes,
            size=10, weight="bold")
plt.savefig("./figures/udm/components.svg")
plt.savefig("./figures/udm/components.eps")
plt.close()

plt.figure(figsize=(6.6, 8.8))
cmap = plt.get_cmap("tab10")
for i, section in enumerate(sections):
    if section in loess_sections:
        facies = "loess"
    elif section in fluvial_sections:
        facies = "fluvial"
    elif section in lake_delta_sections:
        facies = "lake_delta"
    else:
        raise NotImplementedError(section)
    with open(f"./datasets/UDM/{facies}/{section}.udm", "rb") as f:
        udm_result: UDMResult = pickle.load(f)
    dataset: Dataset = udm_result.dataset
    classes = np.expand_dims(np.expand_dims(dataset.classes_phi, axis=0), axis=0).repeat(
        len(dataset), axis=0).repeat(udm_result.n_components, axis=1)
    interval = np.abs((dataset.classes_phi[0] - dataset.classes_phi[-1]) / (dataset.n_classes - 1))
    proportions_udm, components_udm, (m_udm, std_udm, s_udm, k_udm) = GeneralWeibull.interpret(
        udm_result.parameters[-1], classes, interval)
    if facies == "loess":
        xlim = (0.06, 600)
        xticks = (0.1, 1, 10, 100)
    elif facies == "fluvial":
        xlim = (0.06, 2000)
        xticks = (0.1, 1, 10, 100, 1000)
    else:
        xlim = (0.06, 5000)
        xticks = (0.1, 1, 10, 100, 5000)
    axes = plt.subplot(6, 3, i + 1)
    plot_proportions(axes, proportions_udm, title=section)
plt.tight_layout()
for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06,
            f"{string.ascii_uppercase[n]}",
            transform=ax.transAxes,
            size=10, weight="bold")
plt.savefig("./figures/udm/proportions.svg")
plt.savefig("./figures/udm/proportions.eps")
plt.close()
