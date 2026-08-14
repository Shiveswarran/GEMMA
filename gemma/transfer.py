"""
gemma.transfer
--------------
Cross-city transfer by retrieval (Problem 3.2) + evaluation.

Pipeline for a target city c:
  1. Realization map R_c: embed every candidate POI in c into activity space
     (its soft activity distribution q_p and its latent), with a popularity
     prior from visit counts.
  2. For each test trail, embed the prefix -> context summary + predicted
     next-activity distribution (grammar head, optionally blended with a kNN
     retrieval over the source pool, and with the interpretable Markov operator).
  3. Score each candidate POI by activity match + semantic similarity, plus a
     personal-revisit prior (POIs already in the prefix) and, in few-shot,
     a popularity prior estimated from the labelled target trails.
  4. Rank, compute Acc@{1,5,10} / NDCG@{5,10} / MRR (repo NDCG convention),
     and persist the top-K ranked POIs per trail.

zero_shot : source pool only, no popularity prior.
few_shot  : source pool + a few labelled target trails; popularity prior on.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .data import build_city_data, TT_CHECKIN
from .features import temporal_features, spatial_dim
from .utils import get_logger, timer


# --------------------------------------------------------------------------- #
#  Batch helpers                                                              #
# --------------------------------------------------------------------------- #
def _make_ctx_batch(cont_list, cid_list, pid_list, cont_dim, device):
    B = len(cont_list)
    L = max(len(c) for c in cont_list)
    cont = np.zeros((B, L, cont_dim), dtype=np.float32)
    cid = np.zeros((B, L), dtype=np.int64)
    pid = np.zeros((B, L), dtype=np.int64)
    types = np.zeros((B, L), dtype=np.int64)      # all check-ins
    mask = np.zeros((B, L), dtype=np.float32)
    for i in range(B):
        li = len(cont_list[i])
        cont[i, :li] = cont_list[i]
        cid[i, :li] = cid_list[i]
        pid[i, :li] = pid_list[i]
        mask[i, :li] = 1.0
    t = lambda x: torch.as_tensor(x).to(device)
    return {"ctx_cont": t(cont), "ctx_cid": t(cid), "ctx_pid": t(pid),
            "ctx_types": t(types), "ctx_mask": t(mask)}


@torch.no_grad()
def _encode_prefixes(model, city, trail_rows_list, cfg, device, batch=256):
    """Encode prefixes -> (context summaries, predicted next-CATEGORY dists)."""
    summaries, cats = [], []
    for s in range(0, len(trail_rows_list), batch):
        chunk = trail_rows_list[s:s + batch]
        cont_l = [city.cont[r] for r in chunk]
        cid_l = [city.cat_id[r] for r in chunk]
        pid_l = [city.primary_id[r] for r in chunk]
        b = _make_ctx_batch(cont_l, cid_l, pid_l, city.cont_dim, device)
        summary, cat_dist, _ = model.embed_prefix(b)
        summaries.append(summary.cpu().numpy())
        cats.append(cat_dist.cpu().numpy())
    return np.concatenate(summaries), np.concatenate(cats)


def _target_categories(city, rows, n_primary):
    """One-hot next-category vectors for individual target check-ins."""
    oh = np.zeros((len(rows), n_primary), dtype=np.float32)
    for i, r in enumerate(rows):
        oh[i, int(city.primary_id[r])] = 1.0
    return oh


# --------------------------------------------------------------------------- #
#  Pools + candidates                                                         #
# --------------------------------------------------------------------------- #
def build_source_pool(model, cities, cfg, device, logger, max_items=20000):
    """Retrieval pool: (context summary q, observed next-category one-hot)."""
    n_primary = model.n_primary
    Q, NEXT = [], []
    for city in cities:
        rng = np.random.RandomState(cfg.seed)
        trails = city.trails
        if len(trails) > max_items:
            trails = [trails[i] for i in rng.choice(len(trails), max_items, False)]
        prefixes = [t["rows"][:-1] for t in trails if len(t["rows"]) >= 2]
        targets = [int(t["rows"][-1]) for t in trails if len(t["rows"]) >= 2]
        if not prefixes:
            continue
        summ, _ = _encode_prefixes(model, city, prefixes, cfg, device)
        Q.append(summ)
        NEXT.append(_target_categories(city, targets, n_primary))
    Q = np.concatenate(Q) if Q else np.zeros((0, cfg.d_model), np.float32)
    NEXT = np.concatenate(NEXT) if NEXT else np.zeros((0, n_primary), np.float32)
    logger.info(f"source retrieval pool: {len(Q)} prefixes")
    return Q, NEXT


@torch.no_grad()
def build_candidates(model, city, cfg, device):
    """Embed every candidate POI in the city into activity space."""
    pt = city.poi_table
    P = len(pt["venue_ids"])
    name_emb = pt["name_emb"]
    cat_text = pt["cat_text"]
    hour = pt["mean_hour"]
    dow = np.zeros(P, np.float32)
    month = np.zeros(P, np.float32)
    dwell = np.zeros(P, np.float32)
    temporal = temporal_features(hour, dow, month, dwell, cfg.n_time_harmonics)
    spat = np.zeros((P, spatial_dim()), np.float32)
    spat[:, 3] = 1.0                                    # is_first flag
    cont = np.concatenate([name_emb, cat_text, temporal, spat], axis=1).astype(np.float32)

    latents, qpois = [], []
    for s in range(0, P, 512):
        cont_l = [cont[i][None] for i in range(s, min(P, s + 512))]
        cid_l = [pt["cat_id"][i][None] for i in range(s, min(P, s + 512))]
        pid_l = [pt["primary_id"][i][None] for i in range(s, min(P, s + 512))]
        b = _make_ctx_batch(cont_l, cid_l, pid_l, city.cont_dim, device)
        states, summary, _ = model.encode_context(
            b["ctx_cont"], b["ctx_cid"], b["ctx_pid"],
            b["ctx_types"], b["ctx_mask"])
        summary = model.target_norm(summary)
        latents.append(summary.cpu().numpy())
        qpois.append(model.codebook.assign(summary).cpu().numpy())
    latents = np.concatenate(latents)
    qpois = np.concatenate(qpois)
    pop = pt["count"] / pt["count"].sum()
    return {"venue_ids": pt["venue_ids"], "latent": latents, "q": qpois,
            "primary_id": pt["primary_id"], "pop": pop.astype(np.float32),
            "lat": pt["lat"], "lon": pt["lon"],
            "id2row": pt["id2row"]}


# --------------------------------------------------------------------------- #
#  Scoring + ranking                                                          #
# --------------------------------------------------------------------------- #
def _zscore(x, axis=0):
    m = x.mean(axis, keepdims=True)
    s = x.std(axis, keepdims=True) + 1e-6
    return (x - m) / s


def _retrieval_next(summ, pool_Q, pool_NEXT, k):
    """kNN over the source pool -> similarity-weighted next-activity dist."""
    if len(pool_Q) == 0:
        return np.zeros((len(summ), pool_NEXT.shape[1] if pool_NEXT.size else 1),
                        np.float32)
    A = summ / (np.linalg.norm(summ, axis=1, keepdims=True) + 1e-6)
    B = pool_Q / (np.linalg.norm(pool_Q, axis=1, keepdims=True) + 1e-6)
    out = np.zeros((len(summ), pool_NEXT.shape[1]), np.float32)
    for s in range(0, len(summ), 256):
        sims = A[s:s + 256] @ B.T                       # (b, M)
        kk = min(k, sims.shape[1])
        idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r in range(sims.shape[0]):
            w = np.maximum(sims[r, idx[r]], 0.0)
            w = w / (w.sum() + 1e-6)
            out[s + r] = (w[:, None] * pool_NEXT[idx[r]]).sum(0)
    return out


def score_and_rank(summ, next_cat, cand, prefix_vids, cfg,
                   use_popularity=False, query_latlon=None):
    """Return ranked candidate row indices (descending) for one query.

    Activity match = probability the model assigns to each candidate POI's
    (shared) primary category under the predicted next-category grammar.
    A geographic proximity prior (next POI is usually near the last one) is
    the strongest city-agnostic realization signal and is added when the
    query's last-check-in location is known."""
    act_match = next_cat[cand["primary_id"]]             # (P,) transferable
    sem = cand["latent"] @ (summ / (np.linalg.norm(summ) + 1e-6))
    score = ((1 - cfg.sem_blend) * _zscore(act_match)
             + cfg.sem_blend * _zscore(sem))
    if cfg.geo_prior > 0 and query_latlon is not None \
            and np.isfinite(query_latlon).all():
        qlat, qlon = query_latlon
        # equirectangular approximation (monotonic in true distance, fast)
        mlat = np.radians((cand["lat"] + qlat) * 0.5)
        dx = np.radians(cand["lon"] - qlon) * np.cos(mlat)
        dy = np.radians(cand["lat"] - qlat)
        dist_km = 6371.0 * np.sqrt(dx * dx + dy * dy)
        score = score + cfg.geo_prior * _zscore(-np.log1p(dist_km))
    if use_popularity:
        score = score + cfg.popularity_prior * _zscore(cand["pop"])
    if cfg.personal_prior > 0 and prefix_vids:
        bonus = np.zeros(len(cand["venue_ids"]), np.float32)
        for v in prefix_vids:
            r = cand["id2row"].get(int(v))
            if r is not None:
                bonus[r] += cfg.personal_prior
        score = score + bonus
    return np.argsort(-score), score


# --------------------------------------------------------------------------- #
#  Metrics                                                                    #
# --------------------------------------------------------------------------- #
def compute_metrics(ranks: List[int], n_cand: int, eval_ks, ndcg_ks):
    ranks = np.array(ranks)                               # rank of truth (0-based)
    m = {}
    for k in eval_ks:
        m[f"acc@{k}"] = float((ranks < k).mean())
    for k in ndcg_ks:
        hit = ranks < k
        dcg = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)   # repo convention
        m[f"ndcg@{k}"] = float(dcg.mean())
    m["mrr"] = float((1.0 / (ranks + 1)).mean())
    m["n_trails"] = int(len(ranks))
    m["n_candidates"] = int(n_cand)
    return m


# --------------------------------------------------------------------------- #
#  End-to-end city evaluation                                                 #
# --------------------------------------------------------------------------- #
def evaluate_city(model, target_city_data, source_pool, cfg, device, logger,
                  mode="zero_shot", few_shot_pool=None, run_dir="outputs"):
    pool_Q, pool_NEXT = source_pool
    if mode == "few_shot" and few_shot_pool is not None:
        pool_Q = np.concatenate([pool_Q, few_shot_pool[0]])
        pool_NEXT = np.concatenate([pool_NEXT, few_shot_pool[1]])
    use_pop = (mode == "few_shot")

    cand = build_candidates(model, target_city_data, cfg, device)
    n_cand = len(cand["venue_ids"])
    logger.info(f"[{target_city_data.city}/{mode}] candidates={n_cand}")

    eval_trails = [t for t in target_city_data.trails if len(t["rows"]) >= 2]
    prefixes = [t["rows"][:-1] for t in eval_trails]
    truths = [int(target_city_data.venue_id[t["rows"][-1]]) for t in eval_trails]

    summ, grammar_cat = _encode_prefixes(model, target_city_data, prefixes,
                                         cfg, device)
    retr = _retrieval_next(summ, pool_Q, pool_NEXT, cfg.knn_k)
    # blend the model's category grammar head with the kNN source retrieval
    next_cat = ((1 - cfg.retrieval_blend) * grammar_cat
                + cfg.retrieval_blend * retr)
    next_cat = next_cat / (next_cat.sum(1, keepdims=True) + 1e-8)

    lat_arr = target_city_data.df["latitude"].values.astype(np.float32)
    lon_arr = target_city_data.df["longitude"].values.astype(np.float32)

    ranks, rows = [], []
    for i, t in enumerate(eval_trails):
        prefix_vids = [int(target_city_data.venue_id[r]) for r in t["rows"][:-1]]
        last_row = t["rows"][-2]                        # last CONTEXT check-in
        query_latlon = (lat_arr[last_row], lon_arr[last_row])
        order, score = score_and_rank(summ[i], next_cat[i], cand,
                                       prefix_vids, cfg, use_pop,
                                       query_latlon=query_latlon)
        truth = truths[i]
        truth_row = cand["id2row"].get(truth, None)
        if truth_row is None:
            rank = n_cand                                # truth not in vocab
        else:
            rank = int(np.where(order == truth_row)[0][0])
        ranks.append(rank)
        topk = order[:cfg.top_k_save]
        rows.append({
            "trail_id": t["trail_id"], "user_id": t["user_id"],
            "ground_truth_venue_id": truth,
            "rank_of_truth": rank,
            "hit@1": int(rank < 1), "hit@5": int(rank < 5),
            "hit@10": int(rank < 10),
            "top_venue_ids": json.dumps([int(cand["venue_ids"][r]) for r in topk]),
            "top_scores": json.dumps([round(float(score[r]), 4) for r in topk]),
            "pred_primary_category": int(next_cat[i].argmax()),
        })

    metrics = compute_metrics(ranks, n_cand, cfg.eval_ks, cfg.ndcg_ks)
    metrics.update({"city": target_city_data.city, "mode": mode})
    logger.info(f"[{target_city_data.city}/{mode}] "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                           if isinstance(v, float)))

    if cfg.save_predictions:
        pred_path = os.path.join(
            run_dir, f"predictions_{target_city_data.city}_{mode}.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        logger.info(f"saved per-trail predictions -> {pred_path}")
    return metrics


def build_few_shot_pool(model, target_city_train, cfg, device, logger):
    """A few labelled target trails per user, encoded like the source pool."""
    rng = np.random.RandomState(cfg.seed)
    by_user = {}
    for t in target_city_train.trails:
        by_user.setdefault(t["user_id"], []).append(t)
    chosen = []
    for uid, ts in by_user.items():
        k = min(cfg.few_shot_trails_per_user, len(ts))
        idx = rng.choice(len(ts), k, replace=False)
        chosen += [ts[i] for i in idx]
    prefixes = [t["rows"][:-1] for t in chosen if len(t["rows"]) >= 2]
    targets = [int(t["rows"][-1]) for t in chosen if len(t["rows"]) >= 2]
    if not prefixes:
        return None
    summ, _ = _encode_prefixes(model, target_city_train, prefixes, cfg, device)
    cnext = _target_categories(target_city_train, targets, model.n_primary)
    logger.info(f"few-shot pool ({target_city_train.city}): {len(summ)} trails")
    return summ, cnext
