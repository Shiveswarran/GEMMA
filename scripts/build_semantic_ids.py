"""Semantic IDs of places: RQ-VAE trained on the source cities' places, applied to every city.

Each place gets three residual-quantization codes (256 codewords each) computed from its
text embedding, coordinates (relative to the city centre, multi-scale sin/cos), taxonomy and a
coordinate-validity flag, plus a fourth disambiguation code (its index among the places of the
same city that share the first three codes).

python scripts/build_semantic_ids.py --data_root data --fsq_csv <taxonomy.csv> \
    --train_cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
    --all_cities <all 14 cities>
-> sid_cache/<city>_codes.npz {venue_ids, codes}, sid_cache/rqvae.pt
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.config import Config  # noqa: E402
from gemma.data import build_city_data  # noqa: E402
from gemma.semantic_id import RQVAE  # noqa: E402
from gemma.taxonomy import Taxonomy  # noqa: E402
from gemma.text_encoder import TextEncoder  # noqa: E402
from gemma.utils import get_logger, set_seed  # noqa: E402


def place_features(city):
    """Block-balanced feature vector per place.  Places without exact coordinates get a
    deterministic hash of their id in the geometry block so that they stay distinct."""
    pt = city.poi_table
    l2 = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    name = l2(pt["name_emb"].astype(np.float32))
    lat, lon = pt["lat"].astype(np.float32), pt["lon"].astype(np.float32)
    valid = pt["coord_valid"]
    ok = valid > 0.5
    clat = np.nanmean(lat[ok]) if ok.any() else 0.0
    clon = np.nanmean(lon[ok]) if ok.any() else 0.0
    geo = []
    for scale in (1.0, 8.0, 64.0):
        geo += [np.sin((lat - clat) * scale), np.cos((lat - clat) * scale),
                np.sin((lon - clon) * scale), np.cos((lon - clon) * scale)]
    geo = np.stack(geo, 1).astype(np.float32)
    vids = pt["venue_ids"].astype(np.int64)
    hashv = np.stack([np.sin(vids * (k + 1) * 0.9161 + k) for k in range(geo.shape[1])], 1).astype(np.float32)
    geo = l2(np.where(valid[:, None] > 0.5, geo, hashv))
    h = np.full(len(lat), 12.0, np.float32)
    tod = l2(np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], 1).astype(np.float32))
    hier = l2(np.stack([pt["act"], pt["pri"], pt["sec"]], 1).astype(np.float32) + 1.0)
    return np.concatenate([name, geo, 0.5 * tod, 0.5 * hier, valid[:, None].astype(np.float32)],
                          1).astype(np.float32)


def add_disambiguation(codes):
    """Append the index of each place among the places sharing its codes."""
    seen, extra = {}, np.zeros(len(codes), np.int64)
    for i, c in enumerate(map(tuple, codes)):
        extra[i] = seen.get(c, 0); seen[c] = extra[i] + 1
    return np.concatenate([codes, extra[:, None]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--fsq_csv", required=True)
    ap.add_argument("--train_cities", nargs="+", required=True)
    ap.add_argument("--all_cities", nargs="+", required=True)
    ap.add_argument("--poi_emb_cache", default="poi_emb_cache")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--ema", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cache", default="sid_cache")
    a = ap.parse_args()
    set_seed(42)
    os.makedirs(a.cache, exist_ok=True)
    cfg = Config(data_root=a.data_root, fsq_csv=a.fsq_csv, poi_emb_cache=a.poi_emb_cache,
                 use_semantic_id=False)
    logger = get_logger("gemma.sid")
    tax, text_enc = Taxonomy(a.fsq_csv), TextEncoder(cfg, logger)
    cfg.poi_emb_dim = text_enc.dim
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cities = {c: build_city_data(cfg, c, ["train"], tax, text_enc, logger, filter_users=False)
              for c in sorted(set(a.train_cities) | set(a.all_cities))}

    X = torch.as_tensor(np.concatenate([place_features(cities[c]) for c in a.train_cities])).to(dev)
    logger.info(f"RQ-VAE on {len(X)} source places, {X.shape[1]} features")
    model = RQVAE(X.shape[1], d=a.bottleneck, n_levels=a.levels, codebook_size=a.codebook,
                  gamma=a.gamma, ema=a.ema).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.kmeans_init(X[torch.randperm(len(X))[:4096]])
    X_eval = X[torch.randperm(len(X))[:20000]]
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X))
        for s in range(0, len(X), 4096):
            out = model(X[perm[s:s + 4096]])
            opt.zero_grad(); out["loss"].backward(); opt.step()
        model.revive_dead(X[torch.randperm(len(X))[:8192]])
        model.eval()
        codes = model.encode_codes(X_eval).cpu().numpy()
        distinct = len({tuple(c) for c in codes}) / len(codes)      # model selection
        if distinct > best:
            best, best_state = distinct, copy.deepcopy(model.state_dict())
        logger.info(f"epoch {ep}: recon={float(out['recon']):.4f} usage={out['usage']:.2f} "
                    f"distinct={distinct:.3f}")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), os.path.join(a.cache, "rqvae.pt"))
    for c in a.all_cities:
        with torch.no_grad():
            codes = model.encode_codes(torch.as_tensor(place_features(cities[c])).to(dev)).cpu().numpy()
        codes = add_disambiguation(codes.astype(np.int64))
        np.savez(os.path.join(a.cache, f"{c}_codes.npz"), venue_ids=cities[c].poi_table["venue_ids"],
                 codes=codes)
        logger.info(f"[{c}] {len(codes)} places, {len(set(map(tuple, codes[:, :-1])))} distinct 3-level codes")


if __name__ == "__main__":
    main()
