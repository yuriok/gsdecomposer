import argparse
import gc
import os
import pickle
import string
import time
from multiprocessing import Pool, freeze_support, set_start_method
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, wasserstein_distance

from QGrain.statistics import logarithmic, to_phi
from QGrain.models import DistributionType, Sample, SSUResult
from QGrain.ssu import try_ssu

from gsdecomposer import setup_seed
from gsdecomposer.plot_base import *
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.udm.loess import UDM_DATASET_DIR as LOESS_UDM_DATASET_DIR, \
    N_COMPONENTS as LOESS_N_COMPONENTS, TRAIN_SECTIONS as LOESS_TRAIN_SECTIONS
from gsdecomposer.udm.fluvial import UDM_DATASET_DIR as FLUVIAL_UDM_DATASET_DIR, \
    N_COMPONENTS as FLUVIAL_N_COMPONENTS, TRAIN_SECTIONS as FLUVIAL_TRAIN_SECTIONS
from gsdecomposer.udm.lake_delta import UDM_DATASET_DIR as LAKE_DELTA_UDM_DATASET_DIR, \
    N_COMPONENTS as LAKE_DELTA_N_COMPONENTS, TRAIN_SECTIONS as LAKE_DELTA_TRAIN_SECTIONS
from gsdecomposer.gan.dataset import GANDataset
from gsdecomposer.gan.train import ROOT_DIR



def moving_average(x, w):
    return pd.Series(x).rolling(w).mean().to_numpy()


def _get_range(x: np.ndarray):
    x = x[~np.isnan(x)]
    x_01 = np.quantile(x, q=0.05)
    x_99 = np.quantile(x, q=0.95)
    delta = (x_99 - x_01) / 10
    x_min = x_01 - delta
    x_max = x_99 + delta
    return x_min, x_max


def _get_extent(x: np.ndarray, y: np.ndarray):
    return *_get_range(x), *_get_range(y)


def get_statistical_parameters(dataset: Union[UDMDataset, GANDataset]):
    parameters = np.zeros((len(dataset), dataset.n_components + 1, 4))
    classes_phi = to_phi(dataset.classes)
    for i in range(len(dataset)):
        stats = logarithmic(classes_phi, dataset._distributions[i])
        parameters[i, 0, 0] = stats["mean"]
        parameters[i, 0, 1] = stats["std"]
        parameters[i, 0, 2] = stats["skewness"]
        parameters[i, 0, 3] = stats["kurtosis"]
        for j in range(dataset.n_components):
            stats = logarithmic(classes_phi, dataset._components[i, j])
            parameters[i, j + 1, 0] = stats["mean"]
            parameters[i, j + 1, 1] = stats["std"]
            parameters[i, j + 1, 2] = stats["skewness"]
            parameters[i, j + 1, 3] = stats["kurtosis"]
    return parameters


def plot_parameter_histograms(parameters: np.ndarray, save_path: str):
    n_components = parameters.shape[1]
    plt.figure(figsize=(6.6, n_components*1.2))
    for j in range(parameters.shape[1]):
        suffix = "$" if j == 0 else f"_{j}$"
        for k in range(4):
            prefix = (r"$Mz", r"$So", r"$Sk", r"$K")[k]
            plt.subplot(parameters.shape[1], 4, j * 4 + k + 1)
            plt.hist(parameters[:, j, k], bins=200, density=True, histtype="stepfilled", color="grey")
            # plt.xlim(np.quantile(parameters[:, j, k], q=0.001), np.quantile(parameters[:, j, k], q=0.999))
            plt.xlabel(prefix + suffix)
            plt.ylabel("Density")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_density_contours(udm_parameters: np.ndarray, gan_parameters: np.ndarray, save_path: str):
    n_components = udm_parameters.shape[1]
    plt.figure(figsize=(4.4, 3.0 + n_components))
    extent = None
    for i in range(2 * n_components):
        if i % 2 == 0:
            x = udm_parameters[:, i // 2, 0]
            y = udm_parameters[:, i // 2, 1]
            extent = _get_extent(x, y)
        else:
            x = gan_parameters[:, i // 2, 0]
            y = gan_parameters[:, i // 2, 1]
        x_min, x_max, y_min, y_max = extent
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values, bw_method=0.05)
        density = np.reshape(kernel(positions).T, xx.shape)
        axes = plt.subplot(n_components + 1, 2, i + 1)
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        cfset = axes.contourf(xx, yy, density, cmap="coolwarm")
        axes.imshow(np.rot90(density), cmap="coolwarm", extent=[x_min, x_max, y_min, y_max], aspect="auto")
        # cset = axes.contour(xx, yy, density, colors="k", linewidths=0.5)
        # axes.clabel(cset, inline=1, fontsize=6)
        # plt.scatter(x, y, s=1, c="gray", alpha=0.1)
        suffix = "$" if i // 2 == 0 else f"_{i // 2}$"
        axes.set_xlabel(r"$Mz" + suffix)
        axes.set_ylabel(r"$So" + suffix)
    plt.tight_layout()
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(-0.15, 1.06,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_overall_shapes(
        udm_dataset: UDMDataset,
        gan_dataset: GANDataset,
        save_path: str):
    udm_scaled_components = np.repeat(
        np.swapaxes(udm_dataset._proportions, 1, 2),
        udm_dataset.n_classes, 2) * udm_dataset._components
    gan_scaled_components = np.repeat(
        np.swapaxes(gan_dataset._proportions, 1, 2),
        gan_dataset.n_classes, 2) * gan_dataset._components
    plt.figure(figsize=(4.4, 5.0))
    axes = plt.subplot(3, 2, 1)
    plot_gsds(axes, udm_dataset.classes, udm_dataset._distributions)
    axes = plt.subplot(3, 2, 2)
    plot_gsds(axes, gan_dataset.classes, gan_dataset._distributions)
    axes = plt.subplot(3, 2, 3)
    plot_components(axes, udm_dataset.classes, udm_dataset._proportions, udm_scaled_components)
    axes = plt.subplot(3, 2, 4)
    plot_components(axes, gan_dataset.classes, gan_dataset._proportions, gan_scaled_components)
    axes = plt.subplot(3, 2, 5)
    plot_components(axes, udm_dataset.classes, udm_dataset._proportions, udm_dataset._components)
    axes = plt.subplot(3, 2, 6)
    plot_components(axes, gan_dataset.classes, gan_dataset._proportions, gan_dataset._components)
    plt.tight_layout()
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(-0.15, 1.06,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_some_samples(dataset: GANDataset, save_path: str):
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6.6, 8.0))
    for i in range(18):
        plt.subplot(6, 3, i + 1)
        plt.plot(dataset.classes, dataset._distributions[i] * 100,
                 c="#ffffff00", marker=".", ms=3, mfc="k", mec="k")
        plt.plot(dataset.classes, dataset._distributions[i] * 100, c="k")
        for j in range(dataset.n_components):
            plt.plot(dataset.classes, dataset._components[i, j] *
                     dataset._proportions[i, 0, j] * 100, c=cmap(j))
        plt.xscale("log")
        plt.xlim(0.06, 3000)
        plt.ylim(0.0, 10.5)
        if i % 3 == 0:
            plt.ylabel("Frequency (%)")
            plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"])
        else:
            plt.yticks([2, 4, 6, 8], [])
        if i // 3 == 5:
            plt.xlabel(r"Grain size (µm)")
            plt.xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"]),
        else:
            plt.xticks([0.1, 1, 10, 100, 1000], []),
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(0.05, 0.85,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_loss_variations(data_path: str, save_path: str):
    with open(data_path, "rb") as f:
        loss_variations = pickle.load(f)
        x = np.linspace(0, loss_variations["batches_done"], len(loss_variations["d_loss"]))
        plt.figure(figsize=(4.4, 5))
        plt.subplot(2, 1, 1)
        plt.plot(x, loss_variations["g_loss"], color="gray", linewidth=0.4)
        plt.plot(x, moving_average(loss_variations["g_loss"], 100), color="#c02c38", linewidth=0.8)
        plt.xlim(0, loss_variations["batches_done"])
        plt.ylim(*_get_range(loss_variations["g_loss"]))
        xticks, _ = plt.xticks()
        plt.xticks(xticks, [f"{int(tick) // 1000} k" for tick in xticks])
        plt.xlabel("Batch")
        plt.ylabel("Generator loss")
        plt.subplot(2, 1, 2)
        plt.plot(x, loss_variations["d_loss"], color="gray", linewidth=0.4)
        plt.plot(x, moving_average(loss_variations["d_loss"], 100), color="#15559a", linewidth=0.8)
        plt.xlim(0, loss_variations["batches_done"])
        plt.ylim(*_get_range(loss_variations["d_loss"]))
        xticks, _ = plt.xticks()
        plt.xticks(xticks, [f"{int(tick) // 1000} k" for tick in xticks])
        plt.xlabel("Batch")
        plt.ylabel("Discriminator loss")
        plt.tight_layout()
        for n, ax in enumerate(plt.gcf().axes):
            ax.text(-0.15, 1.06,
                    f"{string.ascii_uppercase[n]}",
                    transform=ax.transAxes,
                    size=10, weight="bold")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def get_precision(gan_dataset: GANDataset, n_checks=8):
    x0 = np.array([[210, -135, 145, 1.0],
                   [2.8, 5.5, 2.5, 1.0],
                   [2.4, 3.4, 2.5, 1.0],
                   [2.6, 1.3, 2.3, 1.0],
                   [2.5, -0.9, 1.7, 1.0]], dtype=np.float64).T
    classes_phi = to_phi(gan_dataset.classes)
    batch_precision = []
    for i in range(gan_dataset.n_components):
        component_x0 = np.expand_dims(x0[:, i], axis=1)
        component_precision = []
        for j in range(n_checks):
            sample = Sample(f"S{j + 1}", gan_dataset.classes,
                            classes_phi, gan_dataset._components[j, i, :])
            res, info = try_ssu(sample, DistributionType.GeneralWeibull, 1,
                                x0=component_x0, loss="lmse", optimizer="SLSQP",
                                try_global=False, need_history=False)
            if isinstance(res, SSUResult):
                component_precision.append(-res.loss("lmse"))
            else:
                component_precision.append(np.nan)
        batch_precision.append(component_precision)
    batch_precision = np.array(batch_precision, dtype=np.float64)
    return batch_precision


def get_distance(udm_parameters: np.ndarray, gan_parameters: np.ndarray):
    batch_distance = np.zeros((gan_parameters.shape[1], gan_parameters.shape[2]))
    for j in range(udm_parameters.shape[1]):
        for k in range(udm_parameters.shape[2]):
            batch_distance[j, k] = wasserstein_distance(udm_parameters[:, j, k], gan_parameters[:, j, k])
    return batch_distance


def plot_precision(stats_path: str, save_path: str):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    n_components = stats["precision"].shape[1]
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(4.4, 3.0))
    for i in range(n_components):
        precision = np.array([np.median(values[i][~np.isnan(values[i])]) for values in stats["precision"]])
        plt.plot(stats["batches"], precision, c=cmap(i), label=f"C{(i+1)}")
    plt.xlim(0, stats["batches"][-1])
    xticks, _ = plt.xticks()
    plt.xticks(xticks, [f"{int(tick) // 1000} k" for tick in xticks])
    plt.xlabel("Batch")
    plt.ylabel("Component precision")
    plt.legend(loc="lower right", prop={"size": 6})
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_distance(stats_path: str, save_path: str):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    n_components = stats["distance"].shape[1]
    plt.figure(figsize=(6.6, 5.0))
    cmap = plt.get_cmap("tab10")
    for k in range(stats["distance"].shape[2]):
        title = (r"$Mz$", r"$So$", r"$Sk$", r"$K$")[k]
        plt.subplot(2, 2, k + 1)
        for j in range(n_components):
            plt.plot(stats["batches"], stats["distance"][:, j, k], color="black" if j == 0 else cmap(j-1),
                     linewidth=0.8, label="Overall" if j == 0 else f"C{j}")
        plt.yscale("log")
        plt.xlim(0.0, stats["batches"][-1])
        plt.ylim(0.001, 30)
        xticks, _ = plt.xticks()
        plt.xticks(xticks, [f"{int(tick) // 1000} k" for tick in xticks])
        plt.yticks([0.01, 0.1, 1.0, 10.0], ["0.01", "0.1", "1.0", "10"])
        plt.xlabel("Batch")
        plt.ylabel("Distance")
        plt.title(title)
        if k == 0:
            plt.legend(loc="upper right", prop={"size": 6})
    plt.tight_layout()
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(-0.15, 1.06,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    plt.savefig(save_path)
    plt.close()


def look_checkpoint(args: Tuple[UDMDataset, np.ndarray, str, int, int, bool]):
    udm_dataset, udm_parameters, gan_dataset_dir, batch, seed, fast = args
    setup_seed(seed)
    gan_dataset_path = os.path.join(gan_dataset_dir, "checkpoints", f"{batch}.pkl")
    gan_dataset = GANDataset(gan_dataset_path, size=256 if fast else 1024)
    gan_parameters = get_statistical_parameters(gan_dataset)
    if not fast and batch % 1000 == 0:
        # plot_parameter_histograms(gan_parameters, os.path.join(gan_dataset_dir, "parameter_histograms", f"{batch}.png"))
        plot_density_contours(udm_parameters, gan_parameters, os.path.join(gan_dataset_dir, "density_contours", f"{batch}.png"))
        plot_overall_shapes(udm_dataset, gan_dataset, os.path.join(gan_dataset_dir, "overall_shapes", f"{batch}.png"))
        plot_some_samples(gan_dataset, os.path.join(gan_dataset_dir, "samples", f"{batch}.png"))
        # pass
    n_checks = 8 if fast else 64
    precision = get_precision(gan_dataset, n_checks=n_checks)
    distance = get_distance(udm_parameters, gan_parameters)
    stats = {"precision": precision, "components": gan_dataset._components[:n_checks], "distance": distance}
    del gan_dataset
    del gan_parameters
    gc.collect()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_cpus", type=int, default=2,
                        help="the number of processes used to save figures")
    parser.add_argument("--facies", type=str, default="loess",
                        help="the sedimentary facies to check")
    parser.add_argument("--network_type", type=str, default="mlp_gan",
                        help="the network type of the generator and discriminator "
                             "(mlp_gan, mlp_wgan, mlp_sngan, conv_gan, conv_wgan, conv_sngan)")
    parser.add_argument("-f", "--fast", action="store_true")
    opt = parser.parse_args()

    if opt.facies == "loess":
        udm_dataset_path = os.path.join(LOESS_UDM_DATASET_DIR, "all_datasets.dump")
        n_components = LOESS_N_COMPONENTS
        udm_sections = LOESS_TRAIN_SECTIONS
    elif opt.facies == "fluvial":
        udm_dataset_path = os.path.join(FLUVIAL_UDM_DATASET_DIR, "all_datasets.dump")
        n_components = FLUVIAL_N_COMPONENTS
        udm_sections = FLUVIAL_TRAIN_SECTIONS
    elif opt.facies == "lake_delta":
        udm_dataset_path = os.path.join(LAKE_DELTA_UDM_DATASET_DIR, "all_datasets.dump")
        n_components = LAKE_DELTA_N_COMPONENTS
        udm_sections = LAKE_DELTA_TRAIN_SECTIONS
    else:
        raise NotImplementedError(opt.facies)
    gan_dataset_dir = os.path.join(ROOT_DIR, opt.facies, opt.network_type)

    for path in os.listdir(gan_dataset_dir):
        if path[-4:] == ".png":
            os.remove(os.path.join(gan_dataset_dir, path))

    batches = [int(os.path.splitext(filename)[0]) for filename in
               os.listdir(os.path.join(gan_dataset_dir, "checkpoints"))]
    batches.sort()
    latest_batch = batches[-1] // 10000 * 10000
    if latest_batch == batches[-1]:
        time.sleep(0.1)
    batches = [value for value in batches if value <= latest_batch]
    udm_dataset = UDMDataset(udm_dataset_path, sections=udm_sections, size=1024 if opt.fast else -1)
    udm_parameters = get_statistical_parameters(udm_dataset)
    if opt.fast:
        plot_loss_variations(os.path.join(gan_dataset_dir, "loss_variations.dump"),
                             os.path.join(gan_dataset_dir, "loss_variations.png"))
        gan_dataset_path = os.path.join(gan_dataset_dir, "checkpoints", f"{latest_batch}.pkl")
        gan_dataset = GANDataset(gan_dataset_path, size=64)
        plot_some_samples(gan_dataset, os.path.join(gan_dataset_dir, "samples.png"))
        del gan_dataset
        gc.collect()
    else:
        plot_loss_variations(os.path.join(gan_dataset_dir, "loss_variations.dump"),
                             os.path.join(gan_dataset_dir, "loss_variations.png"))
        gan_dataset_path = os.path.join(gan_dataset_dir, "checkpoints", f"{latest_batch}.pkl")
        gan_dataset = GANDataset(gan_dataset_path, size=8192)
        gan_parameters = get_statistical_parameters(gan_dataset)
        plot_parameter_histograms(gan_parameters, os.path.join(gan_dataset_dir, "parameter_histograms.png"))
        plot_density_contours(udm_parameters, gan_parameters, os.path.join(gan_dataset_dir, "density_contours.png"))
        plot_overall_shapes(udm_dataset, gan_dataset, os.path.join(gan_dataset_dir, "overall_shapes.png"))
        plot_some_samples(gan_dataset, os.path.join(gan_dataset_dir, "samples.png"))
        del gan_dataset
        del gan_parameters
        gc.collect()

    freeze_support()
    stats_path = os.path.join(gan_dataset_dir, "stats.dump")
    params = [(udm_dataset, udm_parameters, gan_dataset_dir, batch, 42, opt.fast) for batch in batches]
    set_start_method("spawn")
    with Pool(opt.n_cpus) as p:
        all_stats = p.map(look_checkpoint, params)
    all_precision = np.zeros((len(batches), *all_stats[0]["precision"].shape))
    all_components = np.zeros((len(batches), *all_stats[0]["components"].shape))
    all_distance = np.zeros((len(batches), *all_stats[0]["distance"].shape))
    for i_batch, batch_stats in enumerate(all_stats):
        all_precision[i_batch] = batch_stats["precision"]
        all_components[i_batch] = batch_stats["components"]
        all_distance[i_batch] = batch_stats["distance"]
    stats_to_save = {"batches": batches, "precision": all_precision,
                     "components": all_components, "distance": all_distance}
    with open(stats_path, "wb") as f:
        pickle.dump(stats_to_save, f)
    if opt.fast:
        plot_precision(stats_path, os.path.join(gan_dataset_dir, "precision.png"))
        plot_distance(stats_path, os.path.join(gan_dataset_dir, "distance.png"))
    else:
        plot_precision(stats_path, os.path.join(gan_dataset_dir, "precision.png"))
        plot_distance(stats_path, os.path.join(gan_dataset_dir, "distance.png"))
