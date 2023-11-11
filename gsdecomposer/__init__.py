import logging
import random
import warnings

import numpy as np
import torch

GRAIN_SIZE_CLASSES = (np.logspace(0, 6, 121) * 0.02)[1:]
N_CLASSES = len(GRAIN_SIZE_CLASSES)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # this setting will raise the time cost of the nets using Conv1d
    # torch.backends.cudnn.deterministic = True
