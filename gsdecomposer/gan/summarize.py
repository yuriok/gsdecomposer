import os
import pickle
import string
from multiprocessing import freeze_support, Pool, set_start_method

import shutil
import numpy as np
import xlwt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from gsdecomposer import setup_seed
from gsdecomposer.plot_base import *
from gsdecomposer.udm.loess import UDM_DATASET_DIR as LOESS_UDM_DATASET_DIR, \
    N_COMPONENTS as LOESS_N_COMPONENTS, TRAIN_SECTIONS as LOESS_TRAIN_SECTIONS
from gsdecomposer.udm.fluvial import UDM_DATASET_DIR as FLUVIAL_UDM_DATASET_DIR, \
    N_COMPONENTS as FLUVIAL_N_COMPONENTS, TRAIN_SECTIONS as FLUVIAL_TRAIN_SECTIONS
from gsdecomposer.udm.lake_delta import UDM_DATASET_DIR as LAKE_DELTA_UDM_DATASET_DIR, \
    N_COMPONENTS as LAKE_DELTA_N_COMPONENTS, TRAIN_SECTIONS as LAKE_DELTA_TRAIN_SECTIONS
from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.gan.train import ROOT_DIR
from gsdecomposer.gan.dataset import GANDataset
from gsdecomposer.gan.check import get_statistical_parameters, _get_extent, get_distance, get_precision


SAVE_DIR = os.path.abspath("./datasets/gan")
networks = ["mlp_gan", "mlp_wgan", "mlp_sngan", "conv_gan", "conv_wgan", "conv_sngan"]
sedimentary_facies = ["loess", "fluvial", "lake_delta"]
selected_batches = {"loess": {"mlp_gan": 500000, "mlp_wgan": 500000, "mlp_sngan": 500000,
                              "conv_gan": 500000, "conv_wgan": 500000, "conv_sngan": 500000},
                    "fluvial": {"mlp_gan": 500000, "mlp_wgan": 500000, "mlp_sngan": 500000,
                                "conv_gan": 500000, "conv_wgan": 500000, "conv_sngan": 500000},
                    "lake_delta": {"mlp_gan": 500000, "mlp_wgan": 500000, "mlp_sngan": 500000,
                                   "conv_gan": 500000, "conv_wgan": 500000, "conv_sngan": 500000}}
n_components_map = {"loess": LOESS_N_COMPONENTS,
                    "fluvial": FLUVIAL_N_COMPONENTS,
                    "lake_delta": LAKE_DELTA_N_COMPONENTS}


def get_gan_stats(args):
    facies, network, udm_parameters= args
    selected_batch = selected_batches[facies][network]
    gan_dataset_path = os.path.join(ROOT_DIR, facies, network, "checkpoints", f"{selected_batch}.pkl")
    gan_dataset = GANDataset(gan_dataset_path, 8192)
    gan_parameters = get_statistical_parameters(gan_dataset)
    precision = get_precision(gan_dataset, n_checks=512)
    distance = get_distance(udm_parameters, gan_parameters)
    stats = {"dataset": gan_dataset, "parameters":
        gan_parameters, "precision": precision, "distance": distance}
    return stats


if __name__ == "__main__":
    setup_seed(42)
    udm_cache = {}
    for facies in sedimentary_facies:
        if facies == "loess":
            udm_dataset_path = os.path.join(LOESS_UDM_DATASET_DIR, "all_datasets.dump")
            sections = LOESS_TRAIN_SECTIONS
        elif facies == "fluvial":
            udm_dataset_path = os.path.join(FLUVIAL_UDM_DATASET_DIR, "all_datasets.dump")
            sections = FLUVIAL_TRAIN_SECTIONS
        elif facies == "lake_delta":
            udm_dataset_path = os.path.join(LAKE_DELTA_UDM_DATASET_DIR, "all_datasets.dump")
            sections = LAKE_DELTA_TRAIN_SECTIONS
        else:
            raise NotImplementedError(facies)
        udm_dataset = UDMDataset(udm_dataset_path, sections, true_gsd=False)
        udm_parameters = get_statistical_parameters(udm_dataset)
        udm_cache[facies] = {"dataset": udm_dataset, "parameters": udm_parameters}

    freeze_support()
    set_start_method("spawn")
    gan_tasks = []
    gan_cache = {}
    for facies in sedimentary_facies:
        for network in networks:
            gan_tasks.append((facies, network, udm_cache[facies]["parameters"]))
    with Pool(len(gan_tasks)) as p:
        all_gan_stats = p.map(get_gan_stats, gan_tasks)
        for (facies, network, _), gan_stats in zip(gan_tasks, all_gan_stats):
            gan_cache[(facies, network)] = gan_stats

    plt.figure(figsize=(6.6, 6.0))
    i_subplot = 1
    for facies in sedimentary_facies:
        precision = {}
        distance = {}
        batches = {}
        for network in networks:
            path = os.path.join(ROOT_DIR, facies, network, "stats.dump")
            with open(path, "rb") as f:
                stats: dict = pickle.load(f)
                batches[network] = stats["batches"]
                precision[network] = np.array([np.median(values[~np.isnan(values)]) for values in stats["precision"]])
                distance[network] = np.array([np.mean(values[:, :2][~np.isnan(values[:, :2])]) for values in stats["distance"]])
        max_batch = max([network_batches[-1] for _, network_batches in batches.items()])

        plt.subplot(3, 2, i_subplot)
        for network in networks:
            plt.plot(batches[network], precision[network], linewidth=0.8,
                     label=network.replace("_", " ").upper())
        plt.xlim(0, max_batch)
        plt.ylim(10.0, 21.0)
        xticks, _ = plt.xticks()
        plt.xticks(xticks, [f"{int(tick) // 1000} k" for tick in xticks])
        plt.xlabel("Batch")
        plt.ylabel("Component precision")
        plt.title(facies.replace("_", " ").capitalize())
        if i_subplot == 1:
            plt.legend(loc="lower right", prop={"size": 6})
        i_subplot += 1

        plt.subplot(3, 2, i_subplot)
        for network in networks:
            plt.plot(batches[network], distance[network], linewidth=0.8, label=network.upper())
        plt.yscale("log")
        plt.xlim(0, max_batch)
        plt.ylim(0.8, 0.004)
        xticks, _ = plt.xticks()
        plt.xticks(xticks, [f"{int(tick) // 1000} k" for tick in xticks])
        plt.yticks([0.1, 0.01], ["0.1", "0.01"])
        plt.xlabel("Batch")
        plt.ylabel("Wasserstein distance")
        plt.title(facies.replace("_", " ").capitalize())
        i_subplot += 1

    plt.tight_layout()
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(-0.15, 1.06,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    os.makedirs("./figures/gan/", exist_ok=True)
    plt.savefig("./figures/gan/history.svg")
    plt.savefig("./figures/gan/history.eps")
    plt.close()


    def plot_density(parameters, i_component, extent):
        x = parameters[:, i_component, 0]
        y = parameters[:, i_component, 1]
        x_min, x_max, y_min, y_max = extent
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values, bw_method=0.05)
        density = np.reshape(kernel(positions).T, xx.shape)
        axes = plt.gca()
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        cfset = axes.contourf(xx, yy, density, cmap="coolwarm")
        axes.imshow(np.rot90(density), cmap="coolwarm", extent=[x_min, x_max, y_min, y_max], aspect="auto")
        # cset = axes.contour(xx, yy, density, colors="k", linewidths=0.5)
        # axes.clabel(cset, inline=1, fontsize=6)
        # plt.scatter(x, y, s=1, c="gray", alpha=0.1)
        suffix = "$" if i_component == 0 else f"_{i_component}$"
        axes.set_xlabel(r"$Mz" + suffix + r" (ϕ)")
        axes.set_ylabel(r"$So" + suffix)


    for i_facies, facies in enumerate(sedimentary_facies):
        n_components = n_components_map[facies]
        udm_parameters = udm_cache[facies]["parameters"]
        plt.figure(figsize=(18, 2 * n_components + 2))
        for i_components in range(n_components + 1):
            extent = _get_extent(udm_parameters[:, i_components, 0], udm_parameters[:, i_components, 1])
            plt.subplot(n_components + 1, len(networks) + 1, i_components * (len(networks) + 1) + 1)
            plot_density(udm_cache[facies]["parameters"], i_components, extent)
            plt.title(f"UDM - {'Overall' if i_components == 0 else f'C{i_components}'}")

        for i_network, network in enumerate(networks):
            gan_parameters = gan_cache[(facies, network)]["parameters"]
            for i_components in range(n_components + 1):
                extent = _get_extent(udm_parameters[:, i_components, 0], udm_parameters[:, i_components, 1])
                plt.subplot(n_components + 1, len(networks) + 1,
                            i_components * (len(networks) + 1) + 2 + i_network)
                plot_density(gan_parameters, i_components, extent)
                plt.title(f"{network.replace('_', ' ').upper()} - {'Overall' if i_components == 0 else f'C{i_components}'}")
        plt.tight_layout()
        plt.savefig(f"./figures/gan/{facies}_density_contours.svg")
        plt.savefig(f"./figures/gan/{facies}_density_contours.eps")
        plt.close()

    plt.figure(figsize=(6.6, 8.8))
    for i_facies, facies in enumerate(sedimentary_facies):
        udm_parameters = udm_cache[facies]["parameters"]
        extent = _get_extent(udm_parameters[:, 0, 0], udm_parameters[:, 0, 1])
        plt.subplot(len(networks) + 1, 3, i_facies + 1)
        plot_density(udm_parameters, 0, extent)
        plt.title(f"{facies.replace('_', ' ').capitalize()} - UDM")
        for i_network, network in enumerate(networks):
            gan_parameters = gan_cache[(facies, network)]["parameters"]
            plt.subplot(len(networks) + 1, 3, i_network * 3 + i_facies + 4)
            plot_density(gan_parameters, 0, extent)
            plt.title(f"{facies.replace('_', ' ').capitalize()} - {network.replace('_', ' ').upper()}")
    plt.tight_layout()
    for n, ax in enumerate(plt.gcf().axes):
        col, row = divmod(n, 7)
        n = row * 3 + col
        ax.text(-0.15, 1.06,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    plt.savefig("./figures/gan/density_contours.svg")
    plt.savefig("./figures/gan/density_contours.eps")
    plt.close()

    for facies in sedimentary_facies:
        os.makedirs(os.path.join(SAVE_DIR, facies), exist_ok=True)
        for network in networks:
            src = os.path.join(ROOT_DIR, facies, network, "checkpoints",
                               f"{selected_batches[facies][network]}.pkl")
            dst = os.path.join(SAVE_DIR, facies, f"{network}.pkl")
            shutil.copy(src, dst)

    workbook = xlwt.Workbook()
    precision_sheet: xlwt.Worksheet = workbook.add_sheet("Component precision")
    precision_sheet.write_merge(0, 1, 0, 0, "Facies")
    precision_sheet.write_merge(0, 1, 1, 1, "Component")
    for i_network, network in enumerate(networks):
        row = 2
        precision_sheet.write_merge(
            0, 0, i_network * 2 + 2, i_network * 2 + 3,
            network.replace("_", " ").upper())
        if row == 2:
            precision_sheet.write(1, i_network * 2 + 2, "Median")
            precision_sheet.write(1, i_network * 2 + 3, "S.D.")

        for facies in sedimentary_facies:
            n_components = n_components_map[facies]
            precision = gan_cache[(facies, network)]["precision"]
            if i_network == 0:
                precision_sheet.write_merge(
                    row, row + n_components, 0, 0,
                    facies.replace("_", " ").capitalize())
            if i_network == 0:
                precision_sheet.write(row, 1, "Overall")
            values = precision
            median = np.median(values[~np.isnan(values)])
            std = np.std(values[~np.isnan(values)])
            precision_sheet.write(row, i_network * 2 + 2, median)
            precision_sheet.write(row, i_network * 2 + 3, std)
            row += 1

            for i_component in range(n_components):
                if i_network == 0:
                    precision_sheet.write(row, 1, f"C{i_component + 1}")
                values = precision[i_component]
                mean = np.mean(values[~np.isnan(values)])
                std = np.std(values[~np.isnan(values)])
                precision_sheet.write(row, i_network * 2 + 2, mean)
                precision_sheet.write(row, i_network * 2 + 3, std)
                row += 1
    distance_sheet: xlwt.Worksheet = workbook.add_sheet("Wasserstein distance")
    distance_sheet.write_merge(0, 1, 0, 0, "Facies")
    distance_sheet.write_merge(0, 1, 1, 1, "Component")
    for i_network, network in enumerate(networks):
        row = 2
        distance_sheet.write_merge(
            0, 0, i_network * 4 + 2, i_network * 4 + 5,
            network.replace("_", " ").upper())
        if row == 2:
            distance_sheet.write(1, i_network * 4 + 2, "Mz")
            distance_sheet.write(1, i_network * 4 + 3, "So")
            distance_sheet.write(1, i_network * 4 + 4, "Sk")
            distance_sheet.write(1, i_network * 4 + 5, "K")
        for facies in sedimentary_facies:
            distance = gan_cache[(facies, network)]["distance"]
            n_rows = distance.shape[0]
            if i_network == 0:
                distance_sheet.write_merge(
                    row, row + n_rows - 1, 0, 0,
                    facies.replace("_", " ").capitalize())
            for i_component in range(n_rows):
                if i_network == 0 and i_component == 0:
                    distance_sheet.write(row, 1, "Overall")
                elif i_network == 0 and i_component != 0:
                    distance_sheet.write(row, 1, f"C{i_component}")
                mz, so, sk, k = distance[i_component]
                distance_sheet.write(row, i_network * 4 + 2, mz)
                distance_sheet.write(row, i_network * 4 + 3, so)
                distance_sheet.write(row, i_network * 4 + 4, sk)
                distance_sheet.write(row, i_network * 4 + 5, k)
                row += 1
    workbook.save("./figures/gan/Stats.xls")
