__all__ = ["DecomposerDataset"]

from typing import *

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from gsdecomposer import GRAIN_SIZE_CLASSES
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.gan.dataset import GANDataset


class DecomposerDataset(Dataset):
    def __init__(self, datasets: List[Union[UDMDataset, GANDataset]], device="numpy"):
        self._classes = GRAIN_SIZE_CLASSES
        self._distributions = []
        self._proportions = []
        self._components = []
        for dataset in datasets:
            dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)
            for distributions, proportions, components in dataloader:
                self._distributions.append(distributions)
                self._proportions.append(proportions)
                self._components.append(components)
        self._distributions = np.concatenate(self._distributions, axis=0)
        self._proportions = np.concatenate(self._proportions, axis=0)
        self._components = np.concatenate(self._components, axis=0)
        if device != "numpy":
            self._distributions = torch.from_numpy(self._distributions).to(device)
            self._proportions = torch.from_numpy(self._proportions).to(device)
            self._components = torch.from_numpy(self._components).to(device)

    @property
    def classes(self) -> np.ndarray:
        return self._classes

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @property
    def n_components(self) -> int:
        return self._components.shape[1]

    def __len__(self):
        return self._distributions.shape[0]

    def __getitem__(self, index):
        distributions = self._distributions[index]
        proportions = self._proportions[index]
        components = self._components[index]
        return distributions, proportions, components
