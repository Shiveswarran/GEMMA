import logging
import os
import random

import numpy as np


def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def get_logger(name, logfile=None):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            "%H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); lg.addHandler(ch)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg
