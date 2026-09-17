"""
gemma.downstream
----------------
GEMMA-as-backbone adaptation + evaluation for the Massive-STEPS tasks.

Adaptation modes (shot):
  zero_shot : pretrained backbone only; no target training. Towers + hierarchy
              heads transfer as-is; rank target-city POIs.
  few_shot  : light fine-tune of query/POI towers + heads on a few target trails.
  full      : train on the city's train+validation (in-city supervised setting).

Next-POI ranker (the requested design), score(candidate) =
    w_tower   * z(  q . p_tower  )                       # user-behaviour two-tower
  + w_hier    * z(  logP(act)+logP(pri)+logP(sec)  )     # predicted hierarchy path
  + w_geo     * z( -log1p(dist(last_checkin, cand)) )    # geography (nearest)
  + w_temporal* z(  temporal_fit )                       # time-of-day match
  + w_revisit *   personal_revisit_bonus                 # user behaviour (revisit)
Candidates are RESTRICTED to POIs whose activity is in the predicted top-k
activities (hier_restrict_topk) -> search only the relevant slice of the vocab.

External SOTA integration uses `ExternalAdapter.export(city)` (contract below).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GEMMA
from .data import build_city_data, temporal_features
from .metrics import classification_metrics, ranking_metrics
from .utils import get_logger


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


# =========================================================================== #
#  External SOTA feature-export contract (GETNext/STHGCN/... wrappers use this)
# =========================================================================== #
class ExternalAdapter:
    """`export(city)` returns the tensors a downstream repo injects at its POI
    embedding table (see README 'Integrating a SOTA model'):
        poi_feat (n_pois, d_tower)  : per-POI GEMMA tower embedding
        poi_ids  (n_pois,)          : venue ids aligned to poi_feat
    A repo-specific wrapper maps its own venue-id ordering to poi_ids and
    concatenates poi_feat into its embedding table (3-line patch)."""
    def __init__(self, backbone, cfg, device):
        self.bb, self.cfg, self.device = backbone, cfg, device

    @torch.no_grad()
    def export(self, city):
        pt = city.poi_table
        name = torch.as_tensor(pt["name_emb"]).to(self.device)
        a = torch.as_tensor(pt["act"]).long().to(self.device)
        p = torch.as_tensor(pt["pri"]).long().to(self.device)
        s = torch.as_tensor(pt["sec"]).long().to(self.device)
        feats = []
        for i in range(0, len(a), 4096):
            feats.append(self.bb.poi_embed(name[i:i+4096], a[i:i+4096],
                                           p[i:i+4096], s[i:i+4096]).cpu().numpy())
        return {"poi_feat": np.concatenate(feats), "poi_ids": pt["venue_ids"]}


# =========================================================================== #
def _prefix_batch(city, rows_list, device):
    B = len(rows_list); L = max(len(r) for r in rows_list); cd = city.cont_dim
    cont = np.zeros((B, L, cd), np.float32)
    act = np.zeros((B, L), np.int64); pri = np.zeros((B, L), np.int64)
    sec = np.zeros((B, L), np.int64); mask = np.zeros((B, L), np.float32)
    for i, rows in enumerate(rows_list):
        for j, r in enumerate(rows):
            cont[i, j] = city.cont[r]; act[i, j] = city.act_id[r]
            pri[i, j] = city.pri_id[r]; sec[i, j] = city.sec_id[r]; mask[i, j] = 1
    t = lambda x: torch.as_tensor(x).to(device)
    return {"cont": t(cont), "act": t(act), "pri": t(pri), "sec": t(sec),
            "types": torch.zeros(B, L, dtype=torch.long, device=device),
            "ctx_mask": t(mask)}


def _candidate_bank(bb, city, device):
    pt = city.poi_table
    name = torch.as_tensor(pt["name_emb"]).to(device)
    a = torch.as_tensor(pt["act"]).long().to(device)
    p = torch.as_tensor(pt["pri"]).long().to(device)
    s = torch.as_tensor(pt["sec"]).long().to(device)
    with torch.no_grad():
        P = []
        for i in range(0, len(a), 4096):
            P.append(bb.poi_embed(name[i:i+4096], a[i:i+4096], p[i:i+4096], s[i:i+4096]).cpu().numpy())
    # candidate mean time-of-day (for temporal fit)
    hours = np.zeros(len(pt["venue_ids"]), np.float32)
    df = city.df
    tod = df["timestamp"].dt.hour.values
    for k, v in enumerate(pt["venue_ids"]):
        sel = df["venue_id"].values == v
        hours[k] = tod[sel].mean() if sel.any() else 12.0
    return {"p": np.concatenate(P), "act": pt["act"], "pri": pt["pri"], "sec": pt["sec"],
            "lat": pt["lat"], "lon": pt["lon"], "hour": hours,
            "venue_ids": pt["venue_ids"], "id2row": pt["id2row"]}


def _few_shot_finetune(bb, city, cfg, device, logger):
    """Light fine-tune of towers + hierarchy heads on a few target trails."""
    import numpy as np
    by_user = {}
    for t in city.trails:
        by_user.setdefault(t["user_id"], []).append(t)
    rng = np.random.RandomState(cfg.seed)
    chosen = []
    for u, ts in by_user.items():
        k = min(cfg.few_shot_trails_per_user, len(ts))
        chosen += [ts[i] for i in rng.choice(len(ts), k, replace=False)]
    chosen = [t for t in chosen if len(t["rows"]) >= 2]
    if not chosen:
        return
    params = (list(bb.query_tower.parameters()) + list(bb.poi_tower.parameters())
              + list(bb.act_head.parameters()) + list(bb.pri_head.parameters())
              + list(bb.sec_head.parameters()) + list(bb.act_emb.parameters())
              + list(bb.pri_emb.parameters()) + list(bb.sec_emb.parameters()))
    opt = torch.optim.AdamW(params, lr=5e-4)
    bb.train()
    for ep in range(cfg.few_shot_epochs):
        rng.shuffle(chosen)
        for s in range(0, len(chosen), 128):
            chunk = chosen[s:s+128]
            b = _prefix_batch(city, [t["rows"][:-1] for t in chunk], device)
            h, summ = bb.extract(b)
            ones = torch.ones(summ.size(0), 1, dtype=torch.long, device=device)
            z1 = bb.tnorm(bb.predictor(summ, ones))[:, 0]
            leaf = [t["rows"][-1] for t in chunk]
            ya = torch.as_tensor([city.act_id[r] for r in leaf]).to(device)
            yp = torch.as_tensor([city.pri_id[r] for r in leaf]).to(device)
            ys = torch.as_tensor([city.sec_id[r] for r in leaf]).to(device)
            la = bb.act_head(z1)
            lp = bb.pri_head(torch.cat([z1, bb.act_emb(ya)], -1))
            ls = bb.sec_head(torch.cat([z1, bb.pri_emb(yp)], -1))
            # tower contrastive on true POI
            name = torch.as_tensor(np.stack([city.name_emb[r] for r in leaf])).to(device)
            q = F.normalize(bb.query_tower(z1), -1)
            ppos = bb.poi_embed(name, ya, yp, ys)
            nce = F.cross_entropy((q @ ppos.t()) / cfg.nce_temp,
                                  torch.arange(len(chunk), device=device))
            loss = F.cross_entropy(la, ya) + F.cross_entropy(lp, yp) + F.cross_entropy(ls, ys) + nce
            opt.zero_grad(); loss.backward(); opt.step()
    bb.eval()
    logger.info(f"[{city.city}] few-shot fine-tuned on {len(chosen)} trails")


def eval_next_poi(bb, city, cfg, device, logger):
    cand = _candidate_bank(bb, city, device)
    eval_trails = [t for t in city.trails if len(t["rows"]) >= 2]
    ranks = []; a_hit = p_hit = s_hit = 0
    lat = city.df["latitude"].values.astype(np.float32)
    lon = city.df["longitude"].values.astype(np.float32)
    l2a = None
    for s0 in range(0, len(eval_trails), 256):
        chunk = eval_trails[s0:s0+256]
        b = _prefix_batch(city, [t["rows"][:-1] for t in chunk], device)
        with torch.no_grad():
            _, act_d, pri_d, sec_d, q = bb.embed_prefix(b)
        act_d = act_d.cpu().numpy(); pri_d = pri_d.cpu().numpy()
        sec_d = sec_d.cpu().numpy(); q = q.cpu().numpy()
        for i, t in enumerate(chunk):
            tower = cand["p"] @ q[i]
            hier = (np.log(act_d[i][cand["act"]] + 1e-8)
                    + np.log(pri_d[i][cand["pri"]] + 1e-8)
                    + np.log(sec_d[i][cand["sec"]] + 1e-8))
            last = t["rows"][-2]
            dlat = cand["lat"] - lat[last]; dlon = cand["lon"] - lon[last]
            dist = np.sqrt(dlat*dlat + dlon*dlon)
            geo = -np.log1p(dist)
            # temporal fit: candidate mean hour vs predicted-next tod (use last hour+delta proxy)
            last_h = city.df["timestamp"].iloc[last].hour
            temporal = -np.abs(((cand["hour"] - last_h + 12) % 24) - 12) / 12.0
            score = (cfg.w_tower * _z(tower) + cfg.w_hier * _z(hier)
                     + cfg.w_geo * _z(geo) + cfg.w_temporal * _z(temporal))
            # hierarchy restriction: keep candidates whose activity in predicted top-k
            topk_act = set(np.argsort(-act_d[i])[:cfg.hier_restrict_topk])
            keep = np.array([a in topk_act for a in cand["act"]])
            if keep.any():
                score = np.where(keep, score, score.min() - 10.0)
            # personal revisit
            prefix_vids = [int(city.venue_id[r]) for r in t["rows"][:-1]]
            for v in prefix_vids:
                r = cand["id2row"].get(v)
                if r is not None:
                    score[r] += cfg.w_revisit
            truth = int(city.venue_id[t["rows"][-1]])
            tr = cand["id2row"].get(truth)
            order = np.argsort(-score)
            rank = len(cand["venue_ids"]) if tr is None else int(np.where(order == tr)[0][0])
            ranks.append(rank)
            tleaf = t["rows"][-1]
            a_hit += int(act_d[i].argmax() == city.act_id[tleaf])
            p_hit += int(pri_d[i].argmax() == city.pri_id[tleaf])
            s_hit += int(sec_d[i].argmax() == city.sec_id[tleaf])
    met = ranking_metrics(ranks)
    N = max(1, len(ranks))
    met.update({"n_trails": len(ranks), "n_candidates": len(cand["venue_ids"]),
                "act@1": a_hit/N, "pri@1": p_hit/N, "sec@1": s_hit/N})
    return met


def eval_next_activity(bb, city, cfg, device):
    eval_trails = [t for t in city.trails if len(t["rows"]) >= 2]
    yt, yp = [], []
    for s0 in range(0, len(eval_trails), 256):
        chunk = eval_trails[s0:s0+256]
        b = _prefix_batch(city, [t["rows"][:-1] for t in chunk], device)
        with torch.no_grad():
            _, act_d, _, _, _ = bb.embed_prefix(b)
        yp += list(act_d.argmax(-1).cpu().numpy())
        yt += [city.act_id[t["rows"][-1]] for t in chunk]
    return classification_metrics(yt, yp)


def run_downstream(bb, fsq, enc, cfg, city_name, run_dir, task="next_poi",
                   shot="zero_shot", logger=None):
    logger = logger or get_logger(f"gemma.down.{city_name}")
    device = cfg.resolved_device()
    splits = ["train", "validation"] if shot == "full" else ["test"]
    if shot == "few_shot":
        train = build_city_data(cfg, city_name, ["train"], fsq, enc, logger, filter_users=False)
        _few_shot_finetune(bb, train, cfg, device, logger)
    if shot == "full":
        train = build_city_data(cfg, city_name, ["train", "validation"], fsq, enc, logger, filter_users=False)
        # reuse few-shot loop with all trails (cheap in-city adaptation of towers/heads)
        cfg2 = cfg; cfg2.few_shot_trails_per_user = 10; cfg2.few_shot_epochs = max(cfg.few_shot_epochs, 5)
        _few_shot_finetune(bb, train, cfg2, device, logger)

    test = build_city_data(cfg, city_name, "test", fsq, enc, logger, filter_users=False)
    if task == "next_poi":
        met = eval_next_poi(bb, test, cfg, device, logger)
    elif task == "next_activity":
        met = eval_next_activity(bb, test, cfg, device)
    else:
        raise ValueError(task)
    met.update({"city": city_name, "task": task, "shot": shot})
    logger.info(f"[{city_name}/{task}/{shot}] " +
                " ".join(f"{k}={round(v,4)}" for k, v in met.items() if isinstance(v, float)))
    with open(os.path.join(run_dir, f"down_{city_name}_{task}_{shot}.json"), "w") as f:
        json.dump(met, f, indent=2)
    return met
