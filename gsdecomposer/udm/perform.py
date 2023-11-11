import os
import pickle

import numpy as np
import tqdm

from QGrain.statistics import to_phi
from QGrain.models import Dataset, UDMResult
from QGrain.distributions import get_distribution
from QGrain.io import load_dataset
from QGrain.udm import try_udm

from gsdecomposer import GRAIN_SIZE_CLASSES
from gsdecomposer.udm.loess import *


if __name__ == "__main__":
    sedimentary_facies = ["loess", "fluvial", "lake_delta"]
    for facies in sedimentary_facies:
        exec(f"from gsdecomposer.udm.{facies} import *", globals(), locals())

        for section in ALL_SECTIONS:
            original_dataset: Dataset = load_dataset(GSD_DATASET_PATH, dataset_name=section,
                                                     sheet_index=SHEET_INDEXES[section], start_col=2)
            distributions = np.zeros((len(original_dataset), len(GRAIN_SIZE_CLASSES)))
            distributions[:, :100] = original_dataset.distributions
            # if len(original_dataset) > 10000:
            #     dataset = Dataset(section, original_dataset.sample_names[49::50], GRAIN_SIZE_CLASSES, distributions[49::50])
            # elif len(original_dataset) > 1000:
            #     dataset = Dataset(section, original_dataset.sample_names[4::5], GRAIN_SIZE_CLASSES, distributions[4::5])
            # else:
            #     dataset = Dataset(section, original_dataset.sample_names, GRAIN_SIZE_CLASSES, distributions)
            dataset = Dataset(section, original_dataset.sample_names, GRAIN_SIZE_CLASSES, distributions)
            x0 = np.array(INITIAL_PARAMETERS[section], dtype=np.float64).T
            total_epochs = UDM_SETTINGS["pretrain_epochs"] + UDM_SETTINGS["max_epochs"]
            with tqdm.tqdm(total=total_epochs) as pbar:
                def update(*args):
                    pbar.set_description(f"Processing {section} section")
                    pbar.update(1)
                udm_result = try_udm(dataset, DISTRIBUTION_TYPE, N_COMPONENTS, x0=x0, **UDM_SETTINGS,
                                     progress_callback=update)
            os.makedirs(UDM_DATASET_DIR, exist_ok=True)
            with open(os.path.join(UDM_DATASET_DIR, f"{section}.udm"), "wb") as f:
                pickle.dump(udm_result, f)

        all_datasets = {}
        classes = GRAIN_SIZE_CLASSES
        classes_phi = to_phi(classes)
        interval_phi = np.abs((classes_phi[0] - classes_phi[-1]) / (len(classes_phi) - 1))

        for i, section in enumerate(ALL_SECTIONS):
            with open(os.path.join(UDM_DATASET_DIR, f"{section}.udm"), "rb") as f:
                udm_result: UDMResult = pickle.load(f)
            classes_for_interpret = np.expand_dims(
                np.expand_dims(classes_phi, axis=0), axis=0).repeat(
                len(udm_result.dataset), axis=0).repeat(udm_result.n_components, axis=1)
            proportions, components, _ = get_distribution(udm_result.distribution_type).interpret(
                udm_result.parameters[-1],
                classes_for_interpret,
                interval_phi)
            udm_dataset = {
                "classes": classes,
                "distributions": udm_result.dataset.distributions,
                "distribution_type": udm_result.distribution_type,
                "parameters": udm_result.parameters[-1]}
            all_datasets[section] = udm_dataset

        with open(os.path.join(UDM_DATASET_DIR, "all_datasets.dump"), "wb") as f:
            pickle.dump(all_datasets, f)
