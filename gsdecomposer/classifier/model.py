import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class Classifier(nn.Module):
    def __init__(self, n_classes: int, n_labels: int, n_features: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.n_labels = n_labels
        self.n_features = n_features
        self.model = nn.Sequential(
            nn.Conv1d(1, n_features, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(n_features, n_features * 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(n_features * 2, n_features * 4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(n_features * 4 * n_classes // 8, n_labels))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)

    def forward(self, x):
        return self.model(x.view(-1, 1, self.n_classes))
