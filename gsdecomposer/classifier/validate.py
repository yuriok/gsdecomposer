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

udm_results = []
labels = []
for label, facies in enumerate(sedimentary_facies):
    for filename in os.listdir(os.path.join(root_dir, facies)):
        if os.path.splitext(filename)[-1] == ".udm":
            udm_results.append(os.path.join(root_dir, facies, filename))
            labels.append(label)
gsd_dataset = ClassifierDataset(udm_results, labels)

train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(
    gsd_dataset, [0.6, 0.2, 0.2], torch.Generator().manual_seed(42))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=64, shuffle=False, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

# net = Classifier(120, len(sedimentary_facies)).to(device)
checkpoint = torch.load(os.path.abspath("../results/classifier.pkl"), map_location=device)
net = checkpoint["model"]
all_predictions = []
all_labels = []
with torch.no_grad():
    for (inputs, labels) in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        all_predictions.append(predictions)
        all_labels.append(labels)
all_predictions = torch.concatenate(all_predictions, dim=0).cpu().numpy()
all_labels = torch.concatenate(all_labels, dim=0).cpu().numpy()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(all_labels, all_predictions)
print(cm)
print(classification_report(all_labels, all_predictions, target_names=sedimentary_facies))
