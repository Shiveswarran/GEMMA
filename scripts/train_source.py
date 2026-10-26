"""Train the mixture (and fine-tune the encoder) on the source cities -> the zero-shot model.

Source training sees only the source cities' training splits, with population statistics
switched off, i.e. exactly the input regime of zero-shot inference.  No data of any target
city is used.

python scripts/train_source.py --run_dir runs/gemma --seed 0
-> runs/gemma/source_s0.pt  {'head', 'encoder', 'cities'}
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.city import CityIndex, sthgcn_filter  # noqa: E402
from gemma.data import build_city_data  # noqa: E402
from gemma.model import new_head  # noqa: E402
from gemma.pipeline import load_encoder, split_ids  # noqa: E402
from gemma.taxonomy import Taxonomy  # noqa: E402
from gemma.text_encoder import TextEncoder  # noqa: E402
from gemma.train import train  # noqa: E402
from gemma.utils import check_device, get_logger, load_config, set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="directory with the pretrained gemma.pt")
    p.add_argument("--data_root"); p.add_argument("--fsq_csv"); p.add_argument("--device")
    p.add_argument("--cities", nargs="+", default=None, help="default: the pretraining cities")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max_queries_per_city", type=int, default=150000)
    p.add_argument("--lr_head", type=float, default=2e-3)
    p.add_argument("--lr_encoder", type=float, default=5e-5, help="0 = frozen encoder")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_per_city", type=int, default=1000)
    p.add_argument("--random_encoder", action="store_true", help="ablation: no pretraining")
    p.add_argument("--out", default=None)
    p.add_argument("--allow_cpu", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.run_dir)
    for k in ("data_root", "fsq_csv", "device"):
        if getattr(a, k):
            setattr(cfg, k, getattr(a, k))
    cfg.seed, cfg.ft_batch, cfg.ft_max_queries = a.seed, a.batch, a.max_queries_per_city
    cfg.val_every, cfg.allow_cpu = a.val_every, a.allow_cpu
    logger = get_logger("gemma.source", os.path.join(a.run_dir, f"train_source_s{a.seed}.log"))
    device = check_device(cfg, logger)
    set_seed(cfg.seed)
    tax, text_enc = Taxonomy(cfg.fsq_csv), TextEncoder(cfg, logger)
    cfg.poi_emb_dim = text_enc.dim
    enc = load_encoder(cfg, a.run_dir, device, random_init=a.random_encoder)

    jobs, vals = [], []
    for c in a.cities or cfg.pretrain_cities:
        city = build_city_data(cfg, c, ["train", "validation"], tax, text_enc, logger, filter_users=False)
        ids = split_ids(city)
        ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device=device)
        q = ci.queries(ids["train"], cfg.ft_all_positions)
        qv = ci.queries(ids["validation"], False)
        if cfg.eval_protocol == "sthgcn":
            qv = sthgcn_filter(ci, qv, ids["train"])
        qv = [qv[i] for i in np.random.RandomState(cfg.seed).permutation(len(qv))[:a.val_per_city]]
        jobs.append((ci, q)); vals.append((ci, qv))
        logger.info(f"[{c}] training queries={len(q)} validation={len(qv)} places={ci.P}")
    head = new_head(enc, jobs[0][0], cfg, device)
    train(enc, head, jobs, cfg, device, logger, a.epochs, a.lr_head, a.lr_encoder, val=vals, tag="source")
    out = a.out or os.path.join(a.run_dir, f"source_s{a.seed}.pt")
    torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                "encoder": {k: v.cpu() for k, v in enc.state_dict().items()},
                "cities": a.cities or cfg.pretrain_cities}, out)
    logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
