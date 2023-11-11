import os

from QGrain.models import KernelType

GSD_DATASET_PATH = os.path.abspath("./datasets/gsd/loess.xlsx")
UDM_DATASET_DIR = os.path.abspath("./datasets/udm/loess")

ALL_SECTIONS = ("BL", "BGY", "GJP", "LC", "TC", "YC", "WN19", "YB19", "FS18", "BSK", "CMG", "NLK", "Osh", "LX")
TRAIN_SECTIONS = ("BL", "BGY", "GJP", "LC", "TC", "YC", "WN19", "YB19")

SHEET_INDEXES = {"BL": 1, "BGY": 2, "GJP": 3, "LC": 4, "TC": 5, "YC": 6, "WN19": 7, "YB19": 8,
                 "FS18": 9, "BSK": 10, "CMG": 11, "NLK": 12, "Osh": 13, "LX": 14}
DISTRIBUTION_TYPE = KernelType.GeneralWeibull
N_COMPONENTS = 3
UDM_SETTINGS = dict(device="cuda",
                    pretrain_epochs=100,
                    min_epochs=200, max_epochs=600,
                    precision=10.0,
                    learning_rate=1e-2,
                    betas=(0.5, 0.999),
                    consider_distance=True,
                    constraint_level=1.25,
                    need_history=False)

INITIAL_PARAMETERS = {"BL": [[210, -135, 145],
                             [2.8, 5.5, 2.5],
                             [2.4, 3.4, 2.5]],
                      "BGY": [[200, -140, 150],
                              [2.4, 5.2, 2.5],
                              [2.5, 2.7, 2.3]],
                      "GJP": [[200, -141, 152],
                              [2.3, 5.3, 2.5],
                              [2.4, 2.6, 2.1]],
                      "LC": [[205, -140, 151],
                             [2.4, 5.9, 2.3],
                             [2.3, 3.4, 2.2]],
                      "TC": [[208, -138, 148],
                             [2.6, 5.5, 2.4],
                             [2.4, 3.4, 2.2]],
                      "YC": [[198, -145, 155],
                             [2.3, 5.3, 2.2],
                             [2.4, 2.8, 2.1]],
                      "WN19": [[211, -137, 147],
                               [2.5, 5.3, 2.8],
                               [2.8, 3.1, 2.3]],
                      "YB19": [[200, -144, 155],
                               [2.3, 5.4, 2.4],
                               [2.4, 2.9, 2.4]],
                      "FS18": [[182, -160, 170],
                               [2.5, 5.4, 2.6],
                               [2.3, 3.0, 2.4]],
                      "BSK": [[161, -170, 180],
                              [2.5, 4.1, 2.8],
                              [2.1, 3.4, 1.1]],
                      "CMG": [[212, -143, 154],
                              [2.6, 6.0, 2.2],
                              [2.8, 3.7, 2.6]],
                      "NLK": [[205, -147, 157],
                              [2.5, 5.7, 2.5],
                              [2.7, 3.0, 2.8]],
                      "Osh": [[213, -143, 154],
                              [2.5, 5.7, 2.5],
                              [2.9, 3.2, 2.9]],
                      "LX": [[203, -132, 143],
                             [2.2, 5.4, 2.5],
                             [2.7, 2.9, 2.4]]}
