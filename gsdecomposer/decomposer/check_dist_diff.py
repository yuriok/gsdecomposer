import os
import string

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from gsdecomposer import setup_seed
from gsdecomposer.plot_base import *
from gsdecomposer.udm.loess import UDM_DATASET_DIR as LOESS_UDM_DATASET_DIR, \
    N_COMPONENTS as LOESS_N_COMPONENTS, TRAIN_SECTIONS as LOESS_TRAIN_SECTIONS

from gsdecomposer.udm.dataset import UDMDataset
from gsdecomposer.gan.dataset import GANDataset
from gsdecomposer.gan.check import get_statistical_parameters, _get_extent
from gsdecomposer.decomposer.dataset import DecomposerDataset


if __name__ == "__main__":
    setup_seed(42)
    udm_dataset_path = os.path.join(LOESS_UDM_DATASET_DIR, "all_datasets.dump")
    clp_sections = LOESS_TRAIN_SECTIONS
    aca_sections = ("BSK", "CMG", "NLK", "Osh")
    clp_udm_dataset = UDMDataset(udm_dataset_path, clp_sections, true_gsd=False)
    clp_udm_parameters = get_statistical_parameters(clp_udm_dataset)
    aca_udm_dataset = UDMDataset(udm_dataset_path, aca_sections, true_gsd=False)
    aca_udm_parameters = get_statistical_parameters(aca_udm_dataset)
    decomposer_datasets = "udm, mlp_gan, mlp_wgan, mlp_sngan, conv_gan, conv_wgan, conv_sngan"
    need_datasets = [text.strip() for text in decomposer_datasets.split(",")]
    _datasets = []
    for dataset_type in need_datasets:
        if dataset_type == "udm":
            udm_dataset = UDMDataset(udm_dataset_path, sections=LOESS_TRAIN_SECTIONS, true_gsd=False, size=4096)
            _datasets.append(udm_dataset)
        else:
            gan_dataset_path = os.path.join("./datasets/gan", "loess", f"{dataset_type}.pkl")
            gan_dataset = GANDataset(gan_dataset_path, 65536)
            _datasets.append(gan_dataset)
    decomposer_dataset = DecomposerDataset(_datasets)
    decomposer_parameters = get_statistical_parameters(decomposer_dataset)
    all_parameters = np.concatenate([clp_udm_parameters, aca_udm_parameters], axis=0)

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
        axes.set_xlabel(r"$Mz" + suffix + " (ϕ)")
        axes.set_ylabel(r"$So" + suffix)
        return cfset

    n_components = LOESS_N_COMPONENTS
    plt.figure(figsize=(6.6, 6.6))
    cfset = None
    for i_components in range(n_components + 1):
        extent = _get_extent(all_parameters[:, i_components, 0], all_parameters[:, i_components, 1])
        plt.subplot(n_components + 1, 3, i_components * 3 + 1)
        plot_density(clp_udm_parameters, i_components, extent)
        plt.title(f"CLP - {'Overall' if i_components == 0 else f'C{i_components}'}")
        plt.subplot(n_components + 1, 3, i_components * 3 + 2)
        plot_density(aca_udm_parameters, i_components, extent)
        plt.title(f"ACA - {'Overall' if i_components == 0 else f'C{i_components}'}")
        plt.subplot(n_components + 1, 3, i_components * 3 + 3)
        cfset = plot_density(decomposer_parameters, i_components, extent)
        plt.title(f"CLP with GANs - {'Overall' if i_components == 0 else f'C{i_components}'}")

    plt.tight_layout()
    for n, ax in enumerate(plt.gcf().axes):
        ax.text(-0.15, 1.06,
                f"{string.ascii_uppercase[n]}",
                transform=ax.transAxes,
                size=10, weight="bold")
    cax = plt.axes((0.1, -0.01, 0.8, 0.01))
    plt.colorbar(cfset, cax=cax, orientation="horizontal", label="Sample density")
    plt.savefig(f"./figures/decomposer/comparison_density_contours.svg")
    plt.savefig(f"./figures/decomposer/comparison_density_contours.eps")
    plt.close()
