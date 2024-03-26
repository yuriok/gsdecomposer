import os
import pickle
import string

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from gsdecomposer.plot_base import *
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.decomposer.train import ROOT_DIR


def _get_range(x: np.ndarray):
    x = x[~np.isnan(x)]
    x_01 = np.quantile(x, q=0.01)
    x_99 = np.quantile(x, q=0.99)
    delta = (x_99 - x_01) / 5
    x_min = x_01 - delta
    x_max = x_99 + delta
    return x_min, x_max


sections = ["GJP", "BGY", "YB19",
            "FS18", "YC", "LC",
            "TC", "WN19", "BL",
            "LX", "BSK", "CMG",
            "NLK", "Osh", "HKG",
            "BS", "HX", "WB1"]

loess_sections = ("GJP", "BGY", "YB19",
                  "FS18", "YC", "LC",
                  "TC", "WN19", "BL",
                  "LX", "BSK", "CMG",
                  "NLK", "Osh")
fluvial_sections = ("HKG", "BS")
lake_delta_sections = ("HX", "WB1")
testing_sections = ("FS18", "LX", "BSK", "CMG", "NLK", "Osh")
experiment_ids = list(range(1, 17))
training_sizes = [512, 1024, 2048, 4096,
                  16384, 28672, 53248, 102400, 200704, 397312,
                  16384, 28672, 53248, 102400, 200704, 397312]
labels = ["512", "1024", "2048", "4096",
          "16 k", "28 k", "52 k", "100 k", "196 k", "388 k",
          "16 k", "28 k", "52 k", "100 k", "196 k", "388 k"]

def m(x, w=100):
    return pd.Series(x).rolling(w).mean().to_numpy()


def lmse(x, y):
    return torch.log(torch.mean(torch.square(x - y), dim=1))


# for facies in ("loess", "fluvial", "lake_delta"):
#     plt.figure(figsize=(6.6, 6.0))
#     cmap = plt.get_cmap("tab10")
#     max_batch = -1
#     for experiment_id in experiment_ids:
#         loss_path = os.path.join(ROOT_DIR, facies, str(experiment_id), "loss_variations.dump")
#         if not os.path.exists(loss_path):
#             continue
#         with open(loss_path, "rb") as f:
#             l = pickle.load(f)
#         if l["batches_done"] > max_batch:
#             max_batch = l["batches_done"]
#         x1 = np.linspace(0, l["batches_done"], len(l["train_loss"]["distributions"]))
#         x2 = np.linspace(0, l["batches_done"], len(l["validate_loss"]["distributions"]))
#         plt.subplot(3, 2, 1)
#         plt.plot(x1, m(l["train_loss"]["proportions"]), color=cmap(experiment_id - 1))
#         plt.subplot(3, 2, 2)
#         plt.plot(x2, m(l["validate_loss"]["proportions"]), color=cmap(experiment_id - 1))
#         plt.subplot(3, 2, 3)
#         plt.plot(x1, m(l["train_loss"]["components"]), color=cmap(experiment_id - 1))
#         plt.subplot(3, 2, 4)
#         plt.plot(x2, m(l["validate_loss"]["components"]), color=cmap(experiment_id - 1))
#         plt.subplot(3, 2, 5)
#         plt.plot(x1, m(l["train_loss"]["distributions"]), color=cmap(experiment_id - 1))
#         plt.subplot(3, 2, 6)
#         plt.plot(x2, m(l["validate_loss"]["distributions"]), color=cmap(experiment_id - 1))
#
#     for i in range(6):
#         plt.subplot(3, 2, i+1)
#         plt.xlim(0, max_batch)
#         plt.xticks(plt.xticks()[0], [f"{int(tick) // 1000} k" for tick in plt.xticks()[0]])
#         plt.xlabel("Batch")
#         row, col = divmod(i, 2)
#         plt.ylabel(("Proportion loss", "Component loss", "Distribution loss")[row])
#         plt.title(("Train", "Validate")[col])
#     plt.tight_layout()
#     plt.show()


plt.figure(figsize=(10.0, 15.0))
i_subplot = 1
for section in sections:
    if section in loess_sections:
        facies = "loess"
    elif section in fluvial_sections:
        facies = "fluvial"
    elif section in lake_delta_sections:
        facies = "lake_delta"
    else:
        raise NotImplementedError(section)
    udm_dataset_path = os.path.join("./datasets/udm/", facies, "all_datasets.dump")
    udm_dataset = UDMDataset(udm_dataset_path, sections=[section], true_gsd=True)
    measured_distributions = torch.from_numpy(udm_dataset._distributions)
    udm_proportions = torch.from_numpy(udm_dataset._proportions)
    udm_components = torch.from_numpy(udm_dataset._components)
    udm_distributions = torch.squeeze(udm_proportions @ udm_components, dim=1)
    udm_error = torch.median(lmse(udm_distributions, measured_distributions))
    errors = []
    for experiment_id in experiment_ids:
        path = os.path.join(ROOT_DIR, facies, str(experiment_id), "checkpoints", "500000.pkl")
        checkpoint = torch.load(path, map_location="cpu")
        decomposer = checkpoint["decomposer"]
        modeled_proportions, modeled_components = decomposer(measured_distributions)
        modeled_distributions = torch.squeeze(modeled_proportions @ modeled_components, dim=1)
        error = lmse(modeled_distributions, measured_distributions)
        errors.append(error.detach().cpu().numpy())
    plt.subplot(6, 3, i_subplot)
    plt.hlines(udm_error, 0, experiment_ids[-1] + 1, colors="red")
    bplot1 = plt.boxplot(errors[:4], labels=labels[:4], positions=experiment_ids[:4], vert=True,
                         patch_artist=True, showfliers=False)
    bplot2 = plt.boxplot(errors[4:-6], labels=labels[4:-6], positions=experiment_ids[4:-6], vert=True,
                         patch_artist=True, showfliers=False)
    bplot3 = plt.boxplot(errors[-6:], labels=labels[4:-6], positions=experiment_ids[4:-6], vert=True,
                         patch_artist=True, showfliers=False)
    for patch in bplot1['boxes']:
        patch.set_facecolor("lightgreen")
    for patch in bplot2['boxes']:
        patch.set_facecolor("pink")
    for patch in bplot3['boxes']:
        patch.set_facecolor("lightblue")
    for bplot in (bplot1, bplot2, bplot3):
        for median in bplot['medians']:
            median.set_color("red")
    plt.xlim(0.5, experiment_ids[:-6][-1] + 0.5)
    # plt.xticks(experiment_ids[:-6], [f"#{i}\n(n={n})" for i, n in zip(experiment_ids[:-6], training_sizes[:-6])], minor=False)
    plt.xticks([], minor=True)
    plt.xlabel("Training set size")
    plt.ylabel("LMSE")
    if section in testing_sections:
        plt.title(section, weight="bold")
    else:
        plt.title(section)
    if i_subplot == 1:
        plt.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]],
                   ["No generator", "One generator", "All generators"],
                   loc="upper right", prop={"size": 6})
    i_subplot += 1

plt.tight_layout()
for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06,
            f"{string.ascii_uppercase[n]}",
            transform=ax.transAxes,
            size=10, weight="bold")
# plt.show()
plt.savefig(f"./figures/decomposer/performance_detailed.svg")
plt.savefig(f"./figures/decomposer/performance_detailed.eps")
