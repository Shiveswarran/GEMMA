from __future__ import annotations

import json
import logging
import os
import random

import numpy as np
import torch

from .config import Config


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_logger(name, logfile=None):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); lg.addHandler(ch)
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        fh = logging.FileHandler(logfile); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def check_device(cfg, logger):
    """Return the device; refuse to train on CPU unless --allow_cpu is given."""
    dev = cfg.resolved_device()
    if dev.startswith("cuda"):
        logger.info(f"device={dev} [{torch.cuda.get_device_name(0)}]")
        return dev
    msg = (f"no GPU visible to torch (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}).")
    if not cfg.allow_cpu:
        raise SystemExit(msg + " Pass --allow_cpu to run on CPU.")
    logger.warning(msg + " Running on CPU.")
    return dev


def amp_dtype(cfg, device):
    """bfloat16 on Ampere or newer, float16 (with loss scaling) on older GPUs, None on CPU."""
    if not (cfg.amp and str(device).startswith("cuda")):
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_config(run_dir) -> Config:
    return Config.from_json(os.path.join(run_dir, "config.json"))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
