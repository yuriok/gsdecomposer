__all__ = ["MLPGenerator", "MLPDiscriminator"]


import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn.utils.parametrizations import spectral_norm


def block(in_features, out_features, dropout=False, normalize=None, relu=True):
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


class ComponentGenerator(nn.Module):
    def __init__(self, latent_dim: int, n_components: int, n_classes: int, n_features: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.n_features = n_features
        self.model = nn.Sequential(
            *block(latent_dim, n_components * n_features),
            *block(n_components * n_features, n_components * n_features * 2),
            *block(n_components * n_features * 2, n_components * n_features * 4),
            *block(n_components * n_features * 4, n_components * n_classes, relu=False))
        self.softmax = nn.Softmax(dim=2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                kaiming_normal_(module.weight.data)

    def forward(self, latent: torch.Tensor):
        components = self.softmax(self.model(latent).view(-1, self.n_components, self.n_classes))
        return components


class ConditionalComponentGenerator(nn.Module):
    def __init__(self, latent_dim: int, n_components: int, n_classes: int, n_features: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.n_features = n_features
        self.model = nn.Sequential(
            *block(n_components + latent_dim, n_features),
            *block(n_features, n_features * 2),
            *block(n_features * 2, n_features * 4),
            *block(n_features * 4, n_classes, relu=False),
            nn.Softmax(dim=1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                kaiming_normal_(module.weight.data)

    def forward(self, latent: torch.Tensor):
        components = []
        for i_component in range(self.n_components):
            labels = torch.zeros((latent.shape[0], self.n_components), device=latent.device)
            labels[:, i_component] = 1.0
            inputs = torch.concatenate([labels, latent], dim=1)
            distributions = self.model(inputs)
            components.append(distributions.unsqueeze(dim=1))
        components = torch.concatenate(components, dim=1)
        return components


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim: int, n_components: int,
                 n_classes: int, n_features: int = 128, conditional=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_classes = n_classes
        self.n_features = n_features
        if conditional:
            self.component_block = ConditionalComponentGenerator(
                latent_dim, n_components, n_classes, n_features)
        else:
            self.component_block = ComponentGenerator(
                latent_dim, n_components, n_classes, n_features)
        self.proportion_block = nn.Sequential(
            *block(latent_dim, n_features),
            *block(n_features, n_features * 2),
            *block(n_features * 2, n_features * 4),
            *block(n_features * 4, n_components, relu=False),
            nn.Softmax(dim=1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                kaiming_normal_(module.weight.data)

    def forward(self, latent: torch.Tensor):
        proportions = self.proportion_block(latent).unsqueeze(dim=1)
        components = self.component_block(latent)
        return proportions, components


class ComponentDiscriminator(nn.Module):
    def __init__(self, n_components: int, n_classes: int, n_features: int = 64, spectral=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_components = n_components
        normalize = "spectral" if spectral else None
        self.model = nn.Sequential(
            nn.Flatten(),
            *block(n_components * n_classes, n_components * n_features * 4, normalize=normalize),
            *block(n_components * n_features * 4, n_components * n_features * 2, normalize=normalize),
            *block(n_components * n_features * 2, n_components * n_features, normalize=normalize),
            *block(n_components * n_features, 1, normalize=normalize, relu=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        score = self.model(components)
        return score


class ConditionalComponentDiscriminator(nn.Module):
    def __init__(self, n_components: int, n_classes: int, n_features: int = 64, spectral=False):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_components = n_components
        normalize = "spectral" if spectral else None
        self.model = nn.Sequential(
            *block(n_components + n_classes, n_features * 4, normalize=normalize),
            *block(n_features * 4, n_features * 2, normalize=normalize),
            *block(n_features * 2, n_features, normalize=normalize),
            *block(n_features, 1, normalize=normalize, relu=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        score = 0
        for i_component in range(self.n_components):
            labels = torch.zeros((components.shape[0], self.n_components), device=components.device)
            labels[:, i_component] = 1.0
            inputs = torch.concatenate([labels, components[:, i_component, :]], dim=1)
            score += self.model(inputs)
        return score / self.n_components


class MLPDiscriminator(nn.Module):
    def __init__(self, n_components: int, n_classes: int,
                 n_features: int = 64, spectral=False, conditional=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        normalize = "spectral" if spectral else None
        if conditional:
            self.component_block = ConditionalComponentDiscriminator(
                n_components, n_classes, n_features, spectral=spectral)
        else:
            self.component_block = ComponentDiscriminator(
                n_components, n_classes, n_features, spectral=spectral)
        self.proportion_block = nn.Sequential(
            nn.Flatten(),
            *block(n_components, n_features * 4, normalize=normalize),
            *block(n_features * 4, n_features * 2, normalize=normalize),
            *block(n_features * 2, n_features, normalize=normalize),
            *block(n_features, 1, normalize=normalize, relu=False))
        self.distribution_block = nn.Sequential(
            nn.Flatten(),
            *block(n_classes, n_features * 4, normalize=normalize),
            *block(n_features * 4, n_features * 2, normalize=normalize),
            *block(n_features * 2, n_features, normalize=normalize),
            *block(n_features, 1, normalize=normalize, relu=False))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)

    def forward(self, distributions, proportions, components) -> torch.Tensor:
        score = (self.distribution_block(distributions) +
                 self.proportion_block(proportions) +
                 self.component_block(components)) / 3
        return score


if __name__ == "__main__":
    from torchsummary import summary
    g = MLPGenerator(64, 5, 120, 64, conditional=True)
    g.to("cuda")
    d = MLPDiscriminator(5, 120, 64, conditional=True)
    d.to("cuda")
    summary(g, input_size=[(64,)])
    summary(d, input_size=[(1, 120), (1, 5), (5, 120)])
