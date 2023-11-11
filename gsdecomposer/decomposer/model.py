from typing import *

import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class Decomposer(nn.Module):
    def __init__(self, n_components: int, n_classes: int, n_features: int = 128):
        super(Decomposer, self).__init__()
        self.n_components = n_components
        self.n_classes = n_classes
        self.n_features = n_features
        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_classes, n_classes),
            nn.LeakyReLU(0.2))
        self.component_block = nn.Sequential(
            nn.Conv1d(1, n_features, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(n_features, n_features * 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(n_features * 2, n_features * 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(n_features * 4, n_components, 3, 1, 1),
            nn.Softmax(dim=2))
        self.proportion_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_classes, n_features),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features, n_features * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features * 2, n_features * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(n_features * 4, n_components),
            nn.Softmax(dim=1))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(module.weight.data)

    def forward(self, distributions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        components = self.component_block(self.linear_block(distributions).view(-1, 1, self.n_classes))
        proportions = self.proportion_block(distributions).view(-1, 1, self.n_components)
        return proportions, components


# class Decomposer(nn.Module):
#     def __init__(self, n_components: int, n_classes: int, n_features: int = 64):
#         super().__init__()
#         self.n_components = n_components
#         self.n_classes = n_classes
#         self.mlp_block = nn.Sequential(
#             nn.Linear(n_classes, n_features * 4),
#             nn.LeakyReLU(0.2),
#             nn.Linear(n_features * 4, n_features * 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(n_features * 2, n_features),
#             nn.LeakyReLU(0.2))
#         self.component_block = nn.Sequential(
#             nn.Linear(n_features + n_classes, n_components * n_classes // 4),
#             nn.LeakyReLU(0.2),
#             nn.Linear(n_components * n_classes // 4, n_components * n_classes // 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(n_components * n_classes // 2, n_components * n_classes))
#         self.proportion_block = nn.Sequential(
#             nn.Linear(n_features + n_classes, n_features * 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(n_features * 2, n_features),
#             nn.LeakyReLU(0.2),
#             nn.Linear(n_features, n_components))
#         self.softmax = nn.Softmax(dim=2)
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv1d)):
#                 kaiming_normal_(m.weight.data)
#
#     def forward(self, distributions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         distributions = distributions.view(-1, self.n_classes)
#         processed = self.mlp_block(distributions)
#         linked = torch.concat([distributions, processed], dim=1)
#         components = self.component_block(linked)
#         components = self.softmax(components.view(-1, self.n_components, self.n_classes))
#         proportions = self.proportion_block(linked)
#         proportions = self.softmax(proportions.view(-1, 1, self.n_components))
#         return proportions, components


if __name__ == "__main__":
    from torchsummary import summary
    d = Decomposer(3, 120)
    d.to("cuda")
    summary(d, input_size=[(1, 120)])
