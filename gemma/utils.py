"""
gemma.utils
-----------
Small shared helpers: reproducible seeding, a logger that writes to both the
console and a run log-file, a stopwatch context manager, and haversine
distance (kept identical to the Massive-STEPS repo's implementation).
"""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from math import radians, sin, cos, sqrt, atan2

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_logger(name: str = "gemma", log_file: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Avoid duplicate handlers if called twice.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None and not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.time()
    logger.info(f"[start] {label}")
    yield
    logger.info(f"[done ] {label} ({time.time() - t0:.1f}s)")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * atan2(sqrt(a), sqrt(1 - a))


def bearing_rad(lat1, lon1, lat2, lon2) -> float:
    """Initial compass bearing from point 1 to point 2, in radians."""
    from math import atan2 as _atan2
    lat1, lat2 = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return _atan2(x, y)
