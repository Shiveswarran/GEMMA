"""
gemma.adapter  -- the "A" in GEMMA: a parameter-efficient ADAPTATION method.

Instead of scoring the full POI vocabulary (traditional), GEMMA does hierarchical
constrained decoding + learned multi-factor reranking on a FROZEN backbone:

  1. PRUNE: from the predicted activity (and primary) distributions, keep only
     candidate POIs whose category path is in the model's top-b -> ~full vocab
     shrinks to a few hundred candidates (never a full-vocab softmax).
  2. SCORE each surviving candidate with a learned fusion of every trained signal:
       - AR semantic-ID code log-likelihood         (GNPR-SID generative ranking)
       - hierarchy path log-prob (act+pri+sec)       (transferable grammar)
       - geo proximity to the last check-in          (geography)
       - temporal fit (time-of-day)                  (temporal)
       - personal revisit count                      (user behaviour)
       - co-visitation transition prior              (collaborative signal)
       - user activity-profile / motif match         (personalized behaviour)
       - popularity                                   (prior)
  3. The AdaptationHead (a tiny MLP over these features) is the only thing trained
     per city (few/full-shot); the backbone is frozen.  Zero-shot uses fixed,
     equal fusion (no adaptation) -> the zero<few<full ladder is the adaptation.

Co-visitation and revisit are target-city statistics, so they are present in
few/full-shot and absent (zero) in zero-shot -- which is exactly why adaptation
adds lift.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATS = ["ar_ll", "hier", "geo", "temporal", "revisit", "covisit", "pop", "user_act"]


class AdaptationHead(nn.Module):
    def __init__(self, hidden=32, n_feat=len(FEATS)):
        super().__init__()
        self.lin = nn.Parameter(torch.ones(n_feat))          # interpretable linear part
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, feats):                                # feats (..., n_feat)
        return (feats * self.lin).sum(-1) + self.mlp(feats).squeeze(-1)


def build_covisitation(city):
    """POI -> next-POI transition counts from the target-city trails (collab signal)."""
    nxt = defaultdict(lambda: defaultdict(float))
    for t in city.trails:
        rows = t["rows"]
        for a, b in zip(rows[:-1], rows[1:]):
            va = int(city.venue_id[a]); vb = int(city.venue_id[b])
            nxt[va][vb] += 1.0
    return nxt


def user_activity_profile(city, prefix_rows, n_act):
    p = np.zeros(n_act, np.float32)
    for r in prefix_rows:
        p[city.act_id[r]] += 1.0
    s = p.sum()
    return p / s if s > 0 else p


def _z(x):
    x = np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    s = x.std()
    if s < 1e-6:                       # constant column -> no signal, return zeros
        return np.zeros_like(x)
    return np.clip((x - x.mean()) / s, -10.0, 10.0)


def candidate_features(cand, q_idx, act_d, pri_d, sec_d, ar_ll_row,
                       last_lat, last_lon, last_hour, prefix_vids, covis, last_vid,
                       uprof, keep):
    """Assemble the per-candidate feature matrix (K_keep, n_feat) for one query."""
    k = keep
    n = int(k.sum())
    f = np.zeros((n, len(FEATS)), np.float32)
    idx = np.where(k)[0]
    f[:, 0] = _z(ar_ll_row[idx])
    f[:, 1] = _z(np.log(act_d[cand["act"][idx]] + 1e-8)
                 + np.log(pri_d[cand["pri"][idx]] + 1e-8)
                 + np.log(sec_d[cand["sec"][idx]] + 1e-8))
    dist = np.sqrt((cand["lat"][idx] - last_lat) ** 2 + (cand["lon"][idx] - last_lon) ** 2)
    f[:, 2] = _z(-np.log1p(dist))
    f[:, 3] = _z(-np.abs(((cand["hour"][idx] - last_hour + 12) % 24) - 12) / 12.0)
    rev = np.zeros(n, np.float32)
    revset = {}
    for v in prefix_vids:
        revset[v] = revset.get(v, 0) + 1
    for j, r in enumerate(idx):
        rev[j] = revset.get(int(cand["venue_ids"][r]), 0)
    f[:, 4] = rev
    cov = np.zeros(n, np.float32)
    if covis is not None and last_vid in covis:
        d = covis[last_vid]
        for j, r in enumerate(idx):
            cov[j] = d.get(int(cand["venue_ids"][r]), 0.0)
        if cov.max() > 0:
            cov = cov / cov.max()
    f[:, 5] = cov
    f[:, 6] = _z(cand["pop"][idx])
    f[:, 7] = uprof[cand["act"][idx]]
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return f, idx
