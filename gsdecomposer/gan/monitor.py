import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from gsdecomposer import setup_seed
from gsdecomposer.plot_base import *
from gsdecomposer.gan.dataset import GANDataset
from gsdecomposer.gan.train import ROOT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", type=str, default="mlp_gan",
                        help="the network type of the generator and discriminator "
                             "(mlp_gan, mlp_wgan, mlp_sngan, conv_gan, conv_wgan, conv_sngan)")
    parser.add_argument("--facies", type=str, default="loess",
                        help="the sedimentary facies to train")
    parser.add_argument("--batch", type=int, default=-1, help="the target batch to check")
    opt = parser.parse_args()

    checkpoint_dir = os.path.join(ROOT_DIR, opt.facies, opt.network_type, "checkpoints")
    cmap = plt.get_cmap("tab10")
    # plt.ion()
    plt.figure(figsize=(6.6, 3.7125))
    frames_dir = "./tmp/video_1_frames"
    frame_index = 0
    while True:
        # setup_seed(42)
        batches = [int(os.path.splitext(filename)[0]) for filename in os.listdir(checkpoint_dir)]
        batches.sort()
        if opt.batch in batches:
            target_batch = opt.batch
        else:
            target_batch = batches[-2]
        dataset_path = os.path.join(checkpoint_dir, f"{target_batch}.pkl")
        dataset = GANDataset(dataset_path, size=64)
        plt.clf()
        title = f"[Facies - {opt.facies.replace('_', ' ').capitalize()}] " \
                f"[GAN Type - {opt.network_type.replace('_', ' ').upper()}] [Batch - {target_batch}]"
        plt.gcf().canvas.manager.window.setWindowTitle(title)
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.plot(dataset.classes, dataset._distributions[i] * 100,
                     c="#ffffff00", marker=".", ms=3, mfc="k", mec="k")
            plt.plot(dataset.classes, dataset._distributions[i] * 100, c="k")
            for j in range(dataset.n_components):
                plt.plot(dataset.classes, dataset._components[i, j] * dataset._proportions[i, 0, j] * 100, c=cmap(j))
            plt.xscale("log")
            plt.xlim(0.06, 3000)
            plt.ylim(0.0, 10.5)
            if i % 4 == 0:
                plt.ylabel("Frequency (%)")
                plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"])
            else:
                plt.yticks([2, 4, 6, 8], [])
            if i // 4 == 2:
                plt.xlabel(r"Grain size (μm)")
                plt.xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"]),
            else:
                plt.xticks([0.1, 1, 10, 100, 1000], []),

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        os.makedirs(frames_dir, exist_ok=True)
        plt.savefig(os.path.join(frames_dir, f"{frame_index}.png"), dpi=300.0)
        frame_index += 1
        # plt.pause(0.1)
        if frame_index > 300:
            break
