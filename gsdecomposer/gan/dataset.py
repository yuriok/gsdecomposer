__all__ = ["GANDataset"]

import numpy as np
import torch
from torch.utils.data import Dataset

from gsdecomposer import GRAIN_SIZE_CLASSES
from gsdecomposer.gan.mlp import MLPGenerator
from gsdecomposer.gan.conv import ConvGenerator


class GANDataset(Dataset):
    def __init__(self, path: str, size: int):
        assert size > 0
        self._classes = GRAIN_SIZE_CLASSES
        with open(path, "rb") as f:
            cp = torch.load(f, map_location="cpu")
        if cp["options"].network_type in ("mlp_gan", "mlp_wgan", "mlp_sngan"):
            generator = MLPGenerator(cp["latent_dim"], cp["n_components"], cp["n_classes"], cp["options"].ngf)
        elif cp["options"].network_type in ("conv_gan", "conv_wgan", "conv_sngan"):
            generator = ConvGenerator(cp["latent_dim"], cp["n_components"], cp["n_classes"], cp["options"].ngf)
        else:
            raise NotImplementedError(cp["options"].network_type)
        generator.load_state_dict(cp["generator_states"])
        generator.eval()
        with torch.no_grad():
            z = torch.randn((size, cp["latent_dim"]), device="cpu")
            proportions, components = generator(z)
            distributions = torch.squeeze(proportions @ components, dim=1)
            self._distributions = distributions.numpy()
            self._proportions = proportions.numpy()
            self._components = components.numpy()

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
