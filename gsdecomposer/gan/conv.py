import typing

import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn.utils.parametrizations import spectral_norm


def linear(in_features, out_features, dropout=False, normalize=None, relu=True):
    layers = []
    if dropout:
        layers.append(nn.Dropout(0.2, inplace=False))
    if normalize is None:
        layers.append(nn.Linear(in_features, out_features, bias=True))
    elif normalize == "batch":
        layers.append(nn.Linear(in_features, out_features, bias=False))
        layers.append(nn.BatchNorm1d(out_features, momentum=0.1))
    elif normalize == "spectral":
        layers.append(spectral_norm(nn.Linear(in_features, out_features, bias=False)))
    else:
        raise NotImplementedError(normalize)
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=False))
    return layers


def conv(in_channels, out_channels, dropout=False, normalize=None, relu=True):
    layers = []
    if dropout:
        layers.append(nn.Dropout(0.2, inplace=False))
    if normalize is None:
        layers.append(nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=True))
    elif normalize == "batch":
        layers.append(nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels, momentum=0.1))
    elif normalize == "spectral":
        layers.append(spectral_norm(nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False)))
    else:
        raise NotImplementedError(normalize)
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=False))
    return layers


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim: int, n_components: int,
                 n_classes: int, n_features: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.n_features = n_features
        self.linear_block = nn.Sequential(*linear(latent_dim, n_classes))
        self.component_block = nn.Sequential(
            *conv(8, n_features * 4),
            nn.Upsample(scale_factor=2),
            *conv(n_features * 4, n_features * 2),
            nn.Upsample(scale_factor=2),
            *conv(n_features * 2, n_features),
            nn.Upsample(scale_factor=2),
            *conv(n_features, n_components, relu=False),
            nn.Softmax(dim=2))
        self.proportion_block = nn.Sequential(
            *linear(latent_dim, n_features),
            *linear(n_features, n_features * 2),
            *linear(n_features * 2, n_features * 4),
            *linear(n_features * 4, n_components, relu=False),
            nn.Softmax(dim=1))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(module.weight.data)

    def forward(self, latent: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        components = self.component_block(
            self.linear_block(latent).view(-1, 8, self.n_classes // 8))
        proportions = self.proportion_block(latent)
        proportions = proportions.view(-1, 1, self.n_components)
        return proportions, components


class ConvDiscriminator(nn.Module):
    def __init__(self, n_components: int, n_classes: int,
                 n_features: int = 64, spectral=False):
        super().__init__()
        self.n_components = n_components
        self.n_classes = n_classes
        self.n_features = n_features
        normalize = "spectral" if spectral else None
        self.distribution_block = nn.Sequential(
            *conv(1, n_features, normalize=normalize),
            nn.MaxPool1d(kernel_size=2),
            *conv(n_features, n_features * 2, normalize=normalize),
            nn.MaxPool1d(kernel_size=2),
            *conv(n_features * 2, n_features * 4, normalize=normalize),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            *linear(n_features * 4 * n_classes // 8, 1, normalize=normalize, relu=False))
        self.proportion_block = nn.Sequential(
            nn.Flatten(),
            *linear(n_components, n_features * 4, normalize=normalize),
            *linear(n_features * 4, n_features * 2, normalize=normalize),
            *linear(n_features * 2, n_features, normalize=normalize),
            *linear(n_features, 1, normalize=normalize, relu=False))
        self.component_block = nn.Sequential(
            *conv(n_components, n_features, normalize=normalize),
            nn.MaxPool1d(kernel_size=2),
            *conv(n_features, n_features * 2, normalize=normalize),
            nn.MaxPool1d(kernel_size=2),
            *conv(n_features * 2, n_features * 4, normalize=normalize),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            *linear(n_features * 4 * n_classes // 8, 1, normalize=normalize, relu=False))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(module.weight.data)

    def forward(self, distributions: torch.Tensor,
                proportions: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        score = (self.distribution_block(distributions.view(-1, 1, self.n_classes)) +
                 self.proportion_block(proportions) +
                 self.component_block(components)) / 3
        return score


if __name__ == "__main__":
    from torchsummary import summary
    g = ConvGenerator(64, 5, 120, 64)
    g.to("cuda")
    d = ConvDiscriminator(5, 120, 64)
    d.to("cuda")
    summary(g, input_size=[(64,)])
    summary(d, input_size=[(1, 120), (1, 5), (5, 120)])
