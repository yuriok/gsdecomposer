__all__ = ["ClassifierDataset"]

from typing import *

import pickle

import numpy as np
from torch.utils.data import Dataset

from QGrain.models import UDMResult


class ClassifierDataset(Dataset):
    def __init__(self, udm_results: List[str], labels: List[int]):
        self._distributions = []
        self._labels = []
        for path, label in zip(udm_results, labels):
            with open(path, "rb") as f:
                udm_result: UDMResult = pickle.load(f)
                self._distributions.append(udm_result.dataset.distributions)
            self._labels.append(np.full(len(udm_result.dataset), fill_value=label))
        self._distributions = np.concatenate(self._distributions, axis=0, dtype=np.float32)
        self._labels = np.concatenate(self._labels, axis=0, dtype=np.int64)

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        return self._distributions[index], self._labels[index]


if __name__ == "__main__":
    import os

    sedimentary_facies = ["loess", "fluvial", "lake_delta"]
    root_dir = "./datasets/udm"
    udm_results = []
    labels = []
    for label, facies in enumerate(sedimentary_facies):
        for filename in os.listdir(os.path.join(root_dir, facies)):
            udm_results.append(os.path.join(root_dir, facies, filename))
            labels.append(label)
    gsd_dataset = ClassifierDataset(udm_results, labels)
    print(len(gsd_dataset))
