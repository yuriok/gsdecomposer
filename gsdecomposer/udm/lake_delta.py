import os

from QGrain.models import KernelType

GSD_DATASET_PATH = os.path.abspath("./datasets/gsd/lake_delta.xlsx")
UDM_DATASET_DIR = os.path.abspath("./datasets/udm/lake_delta")

ALL_SECTIONS = ("HX", "WB1")
TRAIN_SECTIONS = ("HX", "WB1")

SHEET_INDEXES = {"HX": 0, "WB1": 1}
DISTRIBUTION_TYPE = KernelType.GeneralWeibull
N_COMPONENTS = 5
UDM_SETTINGS = dict(device="cuda",
                    pretrain_epochs=100,
                    min_epochs=200, max_epochs=800,
                    precision=10.0,
                    learning_rate=1e-2,
                    betas=(0.5, 0.999),
                    consider_distance=True,
                    constraint_level=1.5,
                    need_history=False)

INITIAL_PARAMETERS = {"HX": [[245, -210, 220],
                             [2.6, 5.5, 2.4],
                             [2.5, 3.0, 2.4],
                             [2.7, 0.7, 2.0],
                             [1.8, -1.5, 2.3]],
                      "WB1": [[240, -210, 220],
                              [2.4, 5.6, 2.6],
                              [2.9, 3.1, 2.6],
                              [2.1, 0.8, 2.4],
                              [2.1, -1.5, 2.0]]}
