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


loess_sections = ("GJP", "BGY", "YB19",
                  "FS18", "YC", "LC",
                  "TC", "WN19", "BL",
                  "LX", "BSK", "CMG",
                  "NLK", "Osh")
train_sections = ("GJP", "BGY", "YB19", "YC",
                  "LC", "TC", "WN19", "BL")

experiment_ids = [1, 2, 3, 4, 5, 6, 7, 8]


def m(x, w=100):
    return pd.Series(x).rolling(w).mean().to_numpy()


def lmse(x, y):
    return torch.log(torch.mean(torch.square(x - y), dim=1))


all_errors = {}
for section in loess_sections:
    facies = "loess"
    udm_dataset_path = os.path.join("./datasets/udm/", facies, "all_datasets.dump")
    udm_dataset = UDMDataset(udm_dataset_path, sections=[section], true_gsd=True)
    measured_distributions = torch.from_numpy(udm_dataset._distributions)
    udm_proportions = torch.from_numpy(udm_dataset._proportions)
    udm_components = torch.from_numpy(udm_dataset._components)
    udm_distributions = torch.squeeze(udm_proportions @ udm_components, dim=1)
    udm_error = torch.mean(lmse(udm_distributions, measured_distributions))
    section_errors = {}
    section_errors["udm"] = udm_error
    for experiment_id in experiment_ids:
        path = os.path.join(ROOT_DIR, facies, str(experiment_id), "checkpoints", "500000.pkl")
        checkpoint = torch.load(path, map_location="cpu")
        decomposer = checkpoint["decomposer"]
        modeled_proportions, modeled_components = decomposer(measured_distributions)
        modeled_distributions = torch.squeeze(modeled_proportions @ modeled_components, dim=1)
        error = lmse(modeled_distributions, measured_distributions)
        # error = 0
        # for i_component in range(3):
        #     error += lmse(modeled_components[:, i_component, :], udm_components[:, i_component, :]) * modeled_proportions[:, 0, i_component]
        section_errors[experiment_id] = error.detach().cpu().numpy()
    all_errors[section] = section_errors

train_errors = []
for experiment_id in experiment_ids:
    _errors = []
    for section in loess_sections:
        if section in train_sections:
            _errors.append(all_errors[section][experiment_id])
    _errors = np.concatenate(_errors, axis=0)
    train_errors.append(_errors)
validate_errors = []
for experiment_id in experiment_ids:
    _errors = []
    for section in loess_sections:
        if section not in train_sections:
            _errors.append(all_errors[section][experiment_id])
    _errors = np.concatenate(_errors, axis=0)
    validate_errors.append(_errors)

plt.figure(figsize=(4.4, 2.2))
plt.subplot(1, 2, 1)
plt.hlines(np.mean([all_errors[section]["udm"] for section in loess_sections if section in train_sections]),
           0, experiment_ids[-1] + 1, colors="red")
plt.boxplot(train_errors, labels=experiment_ids, showfliers=False)
plt.xlim(0.5, experiment_ids[-1]+0.5)
plt.xticks(experiment_ids, [f"#{i}" for i in experiment_ids], minor=False)
plt.xticks([], minor=True)
plt.xlabel("Experiment")
plt.ylabel("LMSE")
plt.title("Training sets")

plt.subplot(1, 2, 2)
plt.boxplot(validate_errors, labels=experiment_ids, showfliers=False)
plt.hlines(np.mean([all_errors[section]["udm"] for section in loess_sections if section not in train_sections]),
           0, experiment_ids[-1] + 1, colors="red")
plt.xlim(0.5, experiment_ids[-1]+0.5)
plt.xticks(experiment_ids, [f"#{i}" for i in experiment_ids], minor=False)
plt.xticks([], minor=True)
plt.xlabel("Experiment")
plt.ylabel("LMSE")
plt.title("Test sets")

plt.tight_layout()
for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06,
            f"{string.ascii_uppercase[n]}",
            transform=ax.transAxes,
            size=10, weight="bold")
# plt.show()
plt.savefig(f"./figures/decomposer/performance.svg")
plt.savefig(f"./figures/decomposer/performance.eps")
