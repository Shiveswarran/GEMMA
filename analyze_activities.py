"""
analyze_activities.py
---------------------
Interpret the learned activity grammar after pretraining:
  * the strongest entries of the transition operator T (activity -> activity),
  * what each latent prototype "means" (its dominant primary categories),
by re-encoding a city's check-ins and looking at which prototype they fall on.

Usage:
    python analyze_activities.py --run_dir outputs/glimpse --city new_york \
        --data_root data
"""
import argparse
import json
import os

import numpy as np
import torch

from gemma.config import GemmaConfig
from gemma.data import build_category_vocab, build_city_data
from gemma.features import NameEmbedder
from gemma.model import GEMMA
from gemma.utils import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--city", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    logger = get_logger("gemma.analyze")

    cfg = GemmaConfig.from_json(os.path.join(args.run_dir, "config.json"))
    cfg.data_root = args.data_root
    ckpt = torch.load(os.path.join(args.run_dir, "gemma.pt"), map_location="cpu",
                      weights_only=False)
    vocab = build_category_vocab(cfg.data_root,
                                 list(dict.fromkeys(cfg.pretrain_cities
                                                    + cfg.transfer_cities)))
    id2primary = {v: k for k, v in vocab.primary2id.items()}

    model = GEMMA(cfg, ckpt["cont_dim"], ckpt["n_cat"], ckpt["n_primary"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 1) transition operator T (activity -> activity)
    T = np.load(os.path.join(args.run_dir, "transition_matrix.npy"))
    logger.info(f"transition matrix T: {T.shape}, strongest transitions:")
    flat = np.dstack(np.unravel_index(np.argsort(-T, axis=None), T.shape))[0]
    for a, b in flat[:args.top]:
        logger.info(f"   activity {a:>2} -> activity {b:>2}   P={T[a,b]:.3f}")

    # 2) prototype -> category composition
    embedder = NameEmbedder(cfg.name_embedder, cfg.sbert_model, cfg.name_emb_dim,
                            os.path.join(cfg.out_dir, "cache_text"), logger)
    city = build_city_data(cfg.data_root, args.city, "train", vocab, embedder,
                           cfg, logger, filter_users=False)
    rows = np.arange(len(city.cont))
    assigns = []
    with torch.no_grad():
        for s in range(0, len(rows), 512):
            chunk = rows[s:s + 512]
            cont = torch.as_tensor(np.stack([city.cont[r] for r in chunk]))[:, None]
            cid = torch.as_tensor(city.cat_id[chunk])[:, None]
            pid = torch.as_tensor(city.primary_id[chunk])[:, None]
            typ = torch.zeros_like(cid)
            mask = torch.ones_like(cid, dtype=torch.float32)
            states, summ, _ = model.encode_context(cont, cid, pid, typ, mask)
            q = model.codebook.assign(model.target_norm(summ))
            assigns.append(q.argmax(-1).numpy())
    proto = np.concatenate(assigns)
    logger.info(f"\nprototype -> dominant primary categories in {args.city}:")
    for k in range(cfg.n_activities):
        sel = city.primary_id[proto == k]
        if len(sel) == 0:
            continue
        vals, cnts = np.unique(sel, return_counts=True)
        top = vals[np.argsort(-cnts)][:3]
        names = ", ".join(id2primary.get(int(t), "?") for t in top)
        logger.info(f"   proto {k:>2} ({len(sel):>5} check-ins): {names}")


if __name__ == "__main__":
    main()
