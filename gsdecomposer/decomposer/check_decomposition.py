import os
import pickle
import random
import string

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from QGrain.models import Dataset, UDMResult
from QGrain.distributions import GeneralWeibull

from gsdecomposer.udm.loess import UDM_DATASET_DIR as LOESS_UDM_DATASET_DIR, \
    N_COMPONENTS as LOESS_N_COMPONENTS, TRAIN_SECTIONS as LOESS_TRAIN_SECTIONS
from gsdecomposer.udm.fluvial import UDM_DATASET_DIR as FLUVIAL_UDM_DATASET_DIR, \
    N_COMPONENTS as FLUVIAL_N_COMPONENTS, TRAIN_SECTIONS as FLUVIAL_TRAIN_SECTIONS
from gsdecomposer.udm.lake_delta import UDM_DATASET_DIR as LAKE_DELTA_UDM_DATASET_DIR, \
    N_COMPONENTS as LAKE_DELTA_N_COMPONENTS, TRAIN_SECTIONS as LAKE_DELTA_TRAIN_SECTIONS
from gsdecomposer.plot_base import *
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.decomposer.train import ROOT_DIR


loess_sections = ("GJP", "BGY", "YB19",
                  "FS18", "YC", "LC",
                  "TC", "WN19", "BL",
                  "LX", "BSK", "CMG",
                  "NLK", "Osh")
fluvial_sections = ("HKG", "BS")
lake_delta_sections = ("HX", "WB1")

aca_loess_sections = ("BSK", "CMG", "NLK", "Osh")
training_loess_sections = ("GJP", "BGY", "YC", "LC", "TC", "BL")
other_sections = ("WN19", "YB19", "HKG", "BS", "HX", "WB1")
test_loess_sections = ("LX", "NLK", "CMG", "Osh", "BSK", "FS18")
selected_sections = ("LX", "NLK", "BSK", "FS18", "HKG", "HX")
all_sections = training_loess_sections + other_sections + test_loess_sections + selected_sections
experiment_id = 13
filenames = ("performance_training_sets_1", "performance_training_sets_2",
             "performance_test_sets", "performance_selected_sets")
for i, section in enumerate(all_sections):
    if section in loess_sections:
        facies = "loess"
    elif section in fluvial_sections:
        facies = "fluvial"
    elif section in lake_delta_sections:
        facies = "lake_delta"
    else:
        raise NotImplementedError(section)
    if i % 6 == 0:
        plt.figure(figsize=(6.6, 8.0))
        cmap = plt.get_cmap("tab10")
    udm_dataset_path = os.path.join("./datasets/udm/", facies, "all_datasets.dump")
    udm_dataset = UDMDataset(udm_dataset_path, sections=[section], true_gsd=True)
    measured_distributions = torch.from_numpy(udm_dataset._distributions)
    udm_proportions = torch.from_numpy(udm_dataset._proportions)
    udm_components = torch.from_numpy(udm_dataset._components)
    udm_distributions = torch.squeeze(udm_proportions @ udm_components, dim=1)
    udm_proportions = udm_proportions.detach().cpu().numpy()
    udm_components = udm_components.detach().cpu().numpy()
    udm_distributions = udm_distributions.detach().cpu().numpy()
    path = os.path.join(ROOT_DIR, facies, str(experiment_id), "checkpoints", "500000.pkl")
    checkpoint = torch.load(path, map_location="cpu")
    decomposer = checkpoint["decomposer"]
    modeled_proportions, modeled_components = decomposer(measured_distributions)
    modeled_distributions = torch.squeeze(modeled_proportions @ modeled_components, dim=1)
    modeled_proportions = modeled_proportions.detach().cpu().numpy()
    modeled_components = modeled_components.detach().cpu().numpy()
    modeled_distributions = modeled_distributions.detach().cpu().numpy()
    if facies == "loess":
        xlim = (0.06, 600)
        xticks = (0.1, 1, 10, 100)
    elif facies == "fluvial":
        xlim = (0.06, 2000)
        xticks = (0.1, 1, 10, 100, 1000)
    else:
        xlim = (0.06, 5000)
        xticks = (0.1, 1, 10, 100, 5000)

    axes_1 = plt.subplot(6, 4, 1 + (i % 6) * 4)
    plot_components(axes_1, udm_dataset.classes, udm_proportions, udm_components,
                    xlim=xlim, xticks=xticks, title=f"{section} - UDM")
    axes_2 = plt.subplot(6, 4, 2 + (i % 6) * 4)
    plot_proportions(axes_2, udm_proportions, title=f"{section} - UDM")
    axes_3 = plt.subplot(6, 4, 3 + (i % 6) * 4)
    plot_components(axes_3, udm_dataset.classes, modeled_proportions, modeled_components,
                    xlim=xlim, xticks=xticks, title=f"{section} - DL")
    axes_4 = plt.subplot(6, 4, 4 + (i % 6) * 4)
    plot_proportions(axes_4, modeled_proportions, title=f"{section} - DL")

    if i % 6 == 5:
        plt.tight_layout()
        for n, ax in enumerate(plt.gcf().axes):
            ax.text(-0.15, 1.06,
                    f"{string.ascii_uppercase[n]}",
                    transform=ax.transAxes,
                    size=10, weight="bold")
        if i//6 == 0 or i//6 == 2:
            handles = [matplotlib.patches.Rectangle([0, 0], 1, 1, color=cmap(i)) for i in range(3)]
            labels = [f"C{i+1}" for i in range(3)]
            axes_2.legend(handles, labels,
                          loc="lower center", bbox_to_anchor=(1.0, -0.75),
                          ncols=3, prop={"size": 6}, columnspacing=20.0)
        else:
            handles = [matplotlib.patches.Rectangle([0, 0], 1, 1, color=cmap(i)) for i in range(5)]
            labels = [f"C{i + 1}" for i in range(5)]
            axes_2.legend(handles, labels,
                          loc="lower center", bbox_to_anchor=(1.0, -0.75),
                          ncols=5, prop={"size": 6}, columnspacing=10.0)
        os.makedirs("./figures/decomposer", exist_ok=True)
        plt.savefig(f"./figures/decomposer/{filenames[i//6]}.svg")
        plt.savefig(f"./figures/decomposer/{filenames[i//6]}.eps")
        plt.close()


def plot_some_samples(facies: str, dataset: UDMDataset, save_path: str):
    measured_distributions = torch.from_numpy(dataset._distributions)
    udm_proportions = torch.from_numpy(dataset._proportions)
    udm_components = torch.from_numpy(dataset._components)
    udm_distributions = torch.squeeze(udm_proportions @ udm_components, dim=1)
    udm_proportions = udm_proportions.detach().cpu().numpy()
    udm_components = udm_components.detach().cpu().numpy()
    udm_distributions = udm_distributions.detach().cpu().numpy()
    path = os.path.join(ROOT_DIR, facies, str(8), "checkpoints", "500000.pkl")
    checkpoint = torch.load(path, map_location="cpu")
    decomposer = checkpoint["decomposer"]
    modeled_proportions, modeled_components = decomposer(measured_distributions)
    modeled_distributions = torch.squeeze(modeled_proportions @ modeled_components, dim=1)
    modeled_proportions = modeled_proportions.detach().cpu().numpy()
    modeled_components = modeled_components.detach().cpu().numpy()
    modeled_distributions = modeled_distributions.detach().cpu().numpy()
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6.6, 8.0))

    for i in range(18):
        sample_index = random.randint(0, measured_distributions.shape[0]-1)
        plt.subplot(6, 3, i + 1)
        plt.plot(dataset.classes, measured_distributions[sample_index] * 100,
                 c="#ffffff00", marker=".", ms=3, mfc="k", mec="k")
        plt.plot(dataset.classes, modeled_distributions[sample_index] * 100, c="k")
        for j in range(dataset.n_components):
            plt.plot(dataset.classes, modeled_components[sample_index, j] *
                     modeled_proportions[sample_index, 0, j] * 100, c=cmap(j))
        plt.xscale("log")
        plt.xlim(0.06, 3000)
        plt.ylim(0.0, 10.5)
        if i % 3 == 0:
            plt.ylabel("Frequency (%)")
            plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"])
        else:
            plt.yticks([2, 4, 6, 8], [])
        if i // 3 == 5:
            plt.xlabel(r"Grain size (µm)")
            plt.xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"]),
        else:
            plt.xticks([0.1, 1, 10, 100, 1000], []),
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(0.05, 0.85,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


loess_dataset = UDMDataset(os.path.join("./datasets/udm/", "loess", "all_datasets.dump"),
                           sections=LOESS_TRAIN_SECTIONS, true_gsd=True)
fluvial_dataset = UDMDataset(os.path.join("./datasets/udm/", "fluvial", "all_datasets.dump"),
                             sections=FLUVIAL_TRAIN_SECTIONS, true_gsd=True)
lake_delta_dataset = UDMDataset(os.path.join("./datasets/udm/", "lake_delta", "all_datasets.dump"),
                                sections=LAKE_DELTA_TRAIN_SECTIONS, true_gsd=True)
plot_some_samples("loess", loess_dataset, "./figures/decomposer/loess_decomposed_samples.svg")
plot_some_samples("fluvial", fluvial_dataset, "./figures/decomposer/fluvial_decomposed_samples.svg")
plot_some_samples("lake_delta", lake_delta_dataset, "./figures/decomposer/lake_delta_decomposed_samples.svg")
