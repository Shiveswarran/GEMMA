"""Make every script land on a real GPU on single-GPU machines.

Call `pin_gpu()` BEFORE `import torch`.  If CUDA_VISIBLE_DEVICES names a GPU
index that does not exist on this node (e.g. CUDA_VISIBLE_DEVICES=2 on a 1-GPU
workstation), torch would silently fall back to CPU; here it is rewritten to an
existing GPU (0) with a printed notice instead."""
import os
import subprocess


def _n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout
        return sum(1 for l in out.splitlines() if l.startswith("GPU"))
    except Exception:
        return None


def pin_gpu():
    n = _n_gpus()
    if not n:
        return                                   # no nvidia-smi / no GPU: leave as is
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        return                                   # unset -> torch sees all GPUs (uses cuda:0)
    ids = [x.strip() for x in v.split(",") if x.strip()]
    ok = [x for x in ids if x.startswith(("GPU-", "MIG-")) or (x.isdigit() and int(x) < n)]
    if not ok:
        print(f"[GEMMA] CUDA_VISIBLE_DEVICES='{v}' but this node has {n} GPU(s) "
              f"-> using GPU 0", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif ok != ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ok)
