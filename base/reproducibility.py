import torch
import torch.cuda
import torch.backends.cudnn
import random
import numpy as np
from utils import log


def set_reproducible(seed, gpu_mode=True):
    log.info('You have chosen to seed training.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_mode:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info('The seeds of random, NumPy and torch are set to {}'.format(seed))