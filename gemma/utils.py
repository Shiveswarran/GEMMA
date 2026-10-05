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


def check_device(cfg, logger):
    """Resolve the device and FAIL LOUDLY instead of silently training on CPU.

    torch falls back to CPU when CUDA_VISIBLE_DEVICES names a GPU index that does
    not exist in this node/job (e.g. CUDA_VISIBLE_DEVICES=2 on a 1-GPU allocation).
    That makes a GPU job run ~100x slower with no error -- it looks 'hung'."""
    import torch
    dev = cfg.resolved_device()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if dev.startswith("cuda"):
        logger.info(f"device={dev} [{torch.cuda.get_device_name(0)}] | "
                    f"CUDA_VISIBLE_DEVICES={cvd} | GPUs visible to torch={n}")
        return dev
    msg = (f"running on CPU: torch sees {n} GPU(s) with CUDA_VISIBLE_DEVICES={cvd}. "
           f"An index that does not exist on this node/job silently falls back to CPU. "
           f"Check `nvidia-smi` and `python -c \"import torch;print(torch.cuda.device_count())\"`.")
    if not getattr(cfg, "allow_cpu", False):
        raise SystemExit("[GEMMA] " + msg + " Refusing to continue (pass --allow_cpu to force).")
    logger.warning(msg)
    return dev


def amp_dtype_for(device):
    """bf16 only where it is NATIVE (compute capability >= 8.0, Ampere+).
    On Turing (e.g. Quadro RTX 6000, sm_75) torch may report bf16 as
    'supported' via slow emulation -> use fp16 + GradScaler instead."""
    import torch
    if not str(device).startswith("cuda"):
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16
