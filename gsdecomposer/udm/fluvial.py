import os

from QGrain.models import KernelType

GSD_DATASET_PATH = os.path.abspath("./datasets/gsd/fluvial.xlsx")
UDM_DATASET_DIR = os.path.abspath("./datasets/udm/fluvial")

ALL_SECTIONS = ("HKG", "BS")
TRAIN_SECTIONS = ("HKG", "BS")

SHEET_INDEXES = {"HKG": 0, "BS": 1}
DISTRIBUTION_TYPE = KernelType.GeneralWeibull
N_COMPONENTS = 4
UDM_SETTINGS = dict(device="cuda",
                    pretrain_epochs=100,
                    min_epochs=200, max_epochs=800,
                    precision=10.0,
                    learning_rate=1e-2,
                    betas=(0.5, 0.999),
                    consider_distance=True,
                    constraint_level=1.25,
                    need_history=False)

INITIAL_PARAMETERS = {"HKG": [[253, -210, 220],
                              [3.1, 3.1, 4.0],
                              [3.0, 2.0, 2.6],
                              [2.7, 0.2, 2.2]],
                      "BS": [[240, -210, 220],
                             [2.6, 4.1, 3.6],
                             [2.5, 1.3, 2.5],
                             [2.4, -0.4, 2.2]]}
