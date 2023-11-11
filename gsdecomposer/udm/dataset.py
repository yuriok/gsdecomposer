__all__ = ["UDMDataset"]

import copy
from typing import *

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from QGrain.models import DistributionType
from QGrain.distributions import get_distribution
from QGrain.statistics import to_phi
from gsdecomposer import GRAIN_SIZE_CLASSES


def _get_data(distribution_type, parameters):
    n_samples, _, n_components = parameters.shape
    classes = GRAIN_SIZE_CLASSES
    classes_phi = to_phi(classes)
    interval_phi = np.abs((classes_phi[0] - classes_phi[-1]) / (len(classes_phi) - 1))
    classes_for_interpret = np.expand_dims(
        np.expand_dims(classes_phi, axis=0), axis=0).repeat(n_samples, axis=0).repeat(n_components, axis=1)
    data = get_distribution(distribution_type).interpret(
        parameters, classes_for_interpret, interval_phi)
    return data


class UDMDataset(Dataset):
    def __init__(self, path: str, sections="all", true_gsd=False, device="numpy", size=-1):
        with open(path, "rb") as f:
            self._all_datasets: Dict[str, Dict[str, Any]] = pickle.load(f)
            self._classes: np.ndarray = GRAIN_SIZE_CLASSES
            self._true_gsd = true_gsd
            self._distributions = []
            self._proportions = []
            self._components = []
            self._mean = []
            self._std = []
            self._skewness = []
            self._kurtosis = []
            if sections == "all":
                sections = [sections for section, _ in self._all_datasets.items()]
            for section in sections:
                dataset = self._all_datasets[section]
                self._distributions.append(dataset["distributions"])
                proportions, components, (mean, std, skewness, kurtosis) = _get_data(dataset["distribution_type"], dataset["parameters"])
                self._proportions.append(proportions)
                self._components.append(components)
                self._mean.append(mean)
                self._std.append(std)
                self._skewness.append(skewness)
                self._kurtosis.append(kurtosis)
            self._distributions = np.concatenate(self._distributions, axis=0, dtype=np.float32)
            self._proportions = np.concatenate(self._proportions, axis=0, dtype=np.float32)
            self._components = np.concatenate(self._components, axis=0, dtype=np.float32)
            self._mean = np.concatenate(self._mean, axis=0, dtype=np.float32)
            self._std = np.concatenate(self._std, axis=0, dtype=np.float32)
            self._skewness = np.concatenate(self._skewness, axis=0, dtype=np.float32)
            self._kurtosis = np.concatenate(self._kurtosis, axis=0, dtype=np.float32)
            if not true_gsd:
                self._distributions = np.squeeze(self._proportions @ self._components, axis=1)
            if size != -1:
                keys = np.random.choice(len(self._distributions), size=size, replace=False)
                self._distributions = self._distributions[keys]
                self._proportions = self._proportions[keys]
                self._components = self._components[keys]
                self._mean = self._mean[keys]
                self._std = self._std[keys]
                self._skewness = self._skewness[keys]
                self._kurtosis = self._kurtosis[keys]
            if device != "numpy":
                self._distributions = torch.from_numpy(self._distributions).to(device)
                self._proportions = torch.from_numpy(self._proportions).to(device)
                self._components = torch.from_numpy(self._components).to(device)
                self._mean = torch.from_numpy(self._mean).to(device)
                self._std = torch.from_numpy(self._std).to(device)
                self._skewness = torch.from_numpy(self._skewness).to(device)
                self._kurtosis = torch.from_numpy(self._kurtosis).to(device)

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @property
    def n_components(self) -> int:
        return self._components.shape[1]

    @property
    def true_gsd(self) -> bool:
        return self._true_gsd

    def __len__(self) -> int:
        return self._distributions.shape[0]

    def __getitem__(self, index):
        distributions = self._distributions[index]
        proportions = self._proportions[index]
        components = self._components[index]
        return distributions, proportions, components
