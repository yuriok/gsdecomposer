import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gsdecomposer.plot_base import *
from gsdecomposer.classifier.model import Classifier
from gsdecomposer.classifier.dataset import ClassifierDataset


def moving_average(x, w):
    return pd.Series(x).rolling(w).mean().to_numpy()


sedimentary_facies = ["loess", "fluvial", "lake_delta"]
root_dir = "./datasets/udm"
device = "cuda"

# udm_results = []
# labels = []
# for label, facies in enumerate(sedimentary_facies):
#     for filename in os.listdir(os.path.join(root_dir, facies)):
#         if os.path.splitext(filename)[-1] == ".udm":
#             udm_results.append(os.path.join(root_dir, facies, filename))
#             labels.append(label)
# gsd_dataset = ClassifierDataset(udm_results, labels)
#
# train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(
#     gsd_dataset, [0.6, 0.2, 0.2], torch.Generator().manual_seed(42))
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
# validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=8, shuffle=False, drop_last=True)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)
#
# net = Classifier(120, len(sedimentary_facies)).to(device)
# criterion = torch.nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5, 0.999))
#
# loss_series = []
# accuracy = {facies: [] for facies in sedimentary_facies}
# n_epochs = 1000
# for epoch in range(n_epochs):
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_dataloader, 0):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         loss_series.append(loss.item())
#         running_loss += loss.item()
#     print(f"[Epoch {epoch}] [Loss {running_loss / len(train_dataloader):.4f}]")
#
#     correct = {facies: 0 for facies in sedimentary_facies}
#     total = {facies: 0 for facies in sedimentary_facies}
#     with torch.no_grad():
#         for (inputs, labels) in validate_dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = net(inputs)
#             _, predictions = torch.max(outputs, 1)
#             for label, prediction in zip(labels, predictions):
#                 if label == prediction:
#                     correct[sedimentary_facies[label]] += 1
#                 total[sedimentary_facies[label]] += 1
#     for facies in sedimentary_facies:
#         accuracy[facies].append(correct[facies] / total[facies])
#         print(f"    Accuracy for [{facies}] is {correct[facies] / total[facies]:.2%}")
#
# checkpoint = {"model": net, "loss": np.array(loss_series), "accuracy": accuracy}
# torch.save(checkpoint, os.path.abspath("../results/classifier.pkl"))

checkpoint = torch.load(os.path.abspath("../results/classifier.pkl"), map_location=device)
n_epochs = 1000
loss_series = checkpoint["loss"]
accuracy = checkpoint["accuracy"]
plt.figure(figsize=(4.4, 3.3))
plt.subplot(2, 2, 1)
x = np.linspace(0, n_epochs, len(loss_series))
plt.plot(x, loss_series, color="gray", linewidth=0.5)
plt.plot(x, moving_average(loss_series, 50), color="#15559a", linewidth=1.0)
plt.xlim(0, n_epochs)
plt.xlabel("Epoch")
plt.ylabel("Loss")

for i, facies in enumerate(sedimentary_facies):
    plt.subplot(2, 2, 2 + i)
    plt.plot(accuracy[facies])
    plt.xlim(0, n_epochs)
    plt.xlabel("Epoch")
    plt.ylabel(f"Accuracy ({facies.replace('_', ' ')})")

plt.tight_layout()
plt.savefig("./figures/classifier.png")

