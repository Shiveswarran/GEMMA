"""Self-supervised pretraining of the GEMMA encoder on the source cities.

python scripts/pretrain.py --data_root data --fsq_csv <taxonomy.csv> --run_name gemma
-> runs/gemma/{gemma.pt, config.json, pretrain.log}
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.config import Config  # noqa: E402
from gemma.pretrain import pretrain  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    for k in ("data_root", "fsq_csv", "poi_encoder", "poi_emb_cache", "sid_cache", "masking",
              "run_name", "out_dir", "device"):
        p.add_argument("--" + k)
    for k in ("epochs", "batch_size", "d_model", "num_workers", "seed"):
        p.add_argument("--" + k, type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--pretrain_cities", nargs="+")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--allow_cpu", action="store_true")
    a = p.parse_args()
    cfg = Config()
    for k, v in vars(a).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.amp = not a.no_amp
    pretrain(cfg)


if __name__ == "__main__":
    main()
