"""
gemma.data
----------
Massive-STEPS check-in loading, per-user DIARY construction (all of a user's
trails joined with a learned [SEP] token so JEPA has a long-enough sequence to
mask -- the fix for the short-trail problem), city-agnostic spatio-temporal
features, and the SSL masking strategies (ablation axis).

Split protocol matches the Massive-STEPS repo: a trail's LAST check-in is the
target, the rest is the context prefix; ranking is over the city POI vocabulary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .fsq import FSQTaxonomy

REQUIRED = ["trail_id", "user_id", "venue_id", "latitude", "longitude",
            "venue_category", "venue_category_id", "timestamp"]

TT_CHECKIN, TT_SEP, TT_USR = 0, 1, 2          # token types


# --------------------------------------------------------------------------- #
def _path(root, city, split):
    return os.path.join(root, city, f"{city}_checkins_{split}.csv")


def load_city_csv(root, city, splits) -> pd.DataFrame:
    splits = [splits] if isinstance(splits, str) else list(splits)
    frames = []
    for sp in splits:
        p = _path(root, city, sp)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no splits {splits} for {city} under {root}")
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "venue_id"]).reset_index(drop=True)
    return df


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    a_lat, a_lon, b_lat, b_lon = map(radians, [a_lat, a_lon, b_lat, b_lon])
    d_lat, d_lon = b_lat - a_lat, b_lon - a_lon
    h = sin(d_lat / 2) ** 2 + cos(a_lat) * cos(b_lat) * sin(d_lon / 2) ** 2
    return 6371.0 * 2 * atan2(sqrt(h), sqrt(1 - h))


def bearing_rad(a_lat, a_lon, b_lat, b_lon):
    a_lat, b_lat = radians(a_lat), radians(b_lat)
    d_lon = radians(b_lon - a_lon)
    x = sin(d_lon) * cos(b_lat)
    y = cos(a_lat) * sin(b_lat) - sin(a_lat) * cos(b_lat) * cos(d_lon)
    return atan2(x, y)


def _cyclic(v, period, n_harm):
    out = []
    for k in range(1, n_harm + 1):
        out += [np.sin(2 * np.pi * k * v / period), np.cos(2 * np.pi * k * v / period)]
    return np.stack(out, -1).astype(np.float32)          # (N, 2*n_harm)


def temporal_features(ts: pd.Series, n_harm: int) -> np.ndarray:
    tod = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek.astype(float)
    month = ts.dt.month.astype(float)
    return np.concatenate([_cyclic(tod.values, 24, n_harm),
                           _cyclic(dow.values, 7, n_harm),
                           _cyclic(month.values, 12, n_harm)], axis=1)


# --------------------------------------------------------------------------- #
@dataclass
class CityData:
    city: str
    df: pd.DataFrame
    cont: np.ndarray          # (N, cont_dim) city-agnostic features
    name_emb: np.ndarray      # (N, poi_emb_dim) POI-knowledge embedding
    venue_id: np.ndarray
    act_id: np.ndarray        # per-checkin FSQ activity id
    pri_id: np.ndarray
    sec_id: np.ndarray
    trails: List[dict]        # [{user_id, trail_id, rows:[...]}]
    poi_table: Dict           # per-POI static attrs for retrieval / candidates

    @property
    def cont_dim(self):
        return self.cont.shape[1]


def build_city_data(cfg, city, splits, fsq: FSQTaxonomy, poi_encoder,
                    logger=None, filter_users=True) -> CityData:
    df = load_city_csv(cfg.data_root, city, splits)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    lat = df["latitude"].values.astype(float)
    lon = df["longitude"].values.astype(float)
    n_bad = int(np.sum(~np.isfinite(lat)) + np.sum(~np.isfinite(lon)))
    if n_bad and logger:
        logger.warning(f"[{city}] {n_bad} non-finite coords -> zeroed spatial")

    # POI-knowledge embedding (pluggable encoder over name+category text)
    names = (df["venue_category"].astype(str) + " : " +
             df.get("name", pd.Series([""] * len(df))).astype(str)).tolist()
    name_emb = poi_encoder.encode(names)                 # (N, poi_emb_dim)

    # spatio-temporal (relative motion + cyclic time)
    temporal = temporal_features(df["timestamp"], cfg.n_time_harmonics)
    spat = np.zeros((len(df), 4), np.float32)
    trail = df["trail_id"].values
    prev_i, prev_tr = -1, None
    for i in range(len(df)):
        if trail[i] != prev_tr:
            spat[i] = [0, 0, 0, 1]                        # is_first
        else:
            d = haversine_km(lat[prev_i], lon[prev_i], lat[i], lon[i])
            b = bearing_rad(lat[prev_i], lon[prev_i], lat[i], lon[i])
            d = 0.0 if not np.isfinite(d) else d
            b = 0.0 if not np.isfinite(b) else b
            spat[i] = [np.log1p(max(d, 0.0)), np.sin(b), np.cos(b), 0]
        prev_i, prev_tr = i, trail[i]

    cont = np.concatenate([name_emb, temporal, spat], 1).astype(np.float32)
    cont = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)

    cat_id = df["venue_category_id"].values
    act_id = np.array([fsq.activity_id(c) for c in cat_id], np.int64)
    pri_id = np.array([fsq.primary_id(c) for c in cat_id], np.int64)
    sec_id = np.array([fsq.secondary_id(c) for c in cat_id], np.int64)

    # trails (+ optional filtering as in the repo)
    trails = []
    for (uid, tid), g in df.groupby(["user_id", "trail_id"], sort=False):
        rows = g.index.to_numpy()
        if len(rows) < cfg.min_trail_len:
            continue
        trails.append({"user_id": int(uid), "trail_id": int(tid), "rows": rows})
    if filter_users:
        by_user = {}
        for t in trails:
            by_user.setdefault(t["user_id"], []).append(t)
        trails = [t for u, ts in by_user.items() if len(ts) >= cfg.min_user_trails
                  for t in ts]

    poi_table = _poi_table(df, name_emb, act_id, pri_id, sec_id, lat, lon)
    return CityData(city, df, cont, name_emb, df["venue_id"].values.astype(np.int64),
                    act_id, pri_id, sec_id, trails, poi_table)


def _poi_table(df, name_emb, act, pri, sec, lat, lon):
    vids = df["venue_id"].values.astype(np.int64)
    order = np.argsort(vids, kind="stable")
    uniq, start = np.unique(vids[order], return_index=True)
    end = np.r_[start[1:], len(order)]
    P = len(uniq)
    nm = np.zeros((P, name_emb.shape[1]), np.float32)
    a = np.zeros(P, np.int64); p = np.zeros(P, np.int64); s = np.zeros(P, np.int64)
    plat = np.zeros(P, np.float32); plon = np.zeros(P, np.float32)
    cnt = np.zeros(P, np.float32)
    la = np.nan_to_num(lat).astype(np.float32); lo = np.nan_to_num(lon).astype(np.float32)
    for k in range(P):
        sel = order[start[k]:end[k]]
        nm[k] = name_emb[sel].mean(0)
        a[k] = np.bincount(act[sel]).argmax()
        p[k] = np.bincount(pri[sel]).argmax()
        s[k] = np.bincount(sec[sel]).argmax()
        plat[k] = la[sel].mean(); plon[k] = lo[sel].mean(); cnt[k] = len(sel)
    return {"venue_ids": uniq, "name_emb": nm, "act": a, "pri": p, "sec": s,
            "lat": plat, "lon": plon, "count": cnt,
            "id2row": {int(v): i for i, v in enumerate(uniq)}}


# --------------------------------------------------------------------------- #
#  Per-user diaries (join trails with [SEP]) + JEPA windowing dataset          #
# --------------------------------------------------------------------------- #
class DiaryDataset(torch.utils.data.Dataset):
    """Each item = one user's diary (trails joined, capped) with a mask over a
    future block, per the chosen masking strategy."""

    def __init__(self, city: CityData, cfg):
        self.city, self.cfg = city, cfg
        by_user = {}
        for t in city.trails:
            by_user.setdefault(t["user_id"], []).append(t)
        self.diaries = []
        for uid, ts in by_user.items():
            ts = sorted(ts, key=lambda x: x["rows"][0])
            rows, types = [], []
            for j, t in enumerate(ts):
                if j > 0:
                    rows.append(-1); types.append(TT_SEP)     # [SEP]
                for r in t["rows"]:
                    rows.append(int(r)); types.append(TT_CHECKIN)
                if len(rows) >= cfg.max_diary_len:
                    break
            rows = rows[:cfg.max_diary_len]; types = types[:cfg.max_diary_len]
            if sum(1 for x in types if x == TT_CHECKIN) >= 3:
                self.diaries.append((np.array(rows), np.array(types)))

    def __len__(self):
        return len(self.diaries)

    def _mask(self, n_ck, rng):
        """Return boolean mask over positions that are TARGETS (predicted)."""
        m = np.zeros(n_ck, bool)
        strat = self.cfg.masking
        if strat == "causal_next":
            m[-1] = True
        elif strat == "random_span":
            k = max(1, int(self.cfg.mask_ratio * n_ck))
            s = rng.randint(1, max(2, n_ck - k + 1))
            m[s:s + k] = True
        elif strat == "activity_block":
            k = max(1, int(self.cfg.mask_ratio * n_ck))
            m[-k:] = True                                    # last activity block
        else:  # block_future (default JEPA)
            b = min(self.cfg.mask_block_max, max(1, int(self.cfg.mask_ratio * n_ck)))
            cut = rng.randint(1, max(2, n_ck - b + 1))
            m[cut:cut + b] = True
        m[0] = False
        return m

    def __getitem__(self, idx):
        rows, types = self.diaries[idx]
        rng = np.random.RandomState(idx)
        ck_pos = np.where(types == TT_CHECKIN)[0]
        tmask_ck = self._mask(len(ck_pos), rng)
        tmask = np.zeros(len(rows), bool)
        tmask[ck_pos[tmask_ck]] = True
        ctx = ~tmask                                          # context positions
        return rows, types, ctx, tmask


def collate(batch, city: CityData, cfg):
    B = len(batch)
    L = max(len(b[0]) for b in batch)
    cd = city.cont_dim
    cont = np.zeros((B, L, cd), np.float32)
    act = np.zeros((B, L), np.int64); pri = np.zeros((B, L), np.int64)
    sec = np.zeros((B, L), np.int64); typ = np.zeros((B, L), np.int64)
    vid = np.full((B, L), -1, np.int64)
    ctx_mask = np.zeros((B, L), np.float32); tgt_mask = np.zeros((B, L), np.float32)
    for i, (rows, types, ctx, tm) in enumerate(batch):
        for j, r in enumerate(rows):
            typ[i, j] = types[j]
            if r >= 0:
                cont[i, j] = city.cont[r]
                act[i, j] = city.act_id[r]; pri[i, j] = city.pri_id[r]; sec[i, j] = city.sec_id[r]
                vid[i, j] = int(city.venue_id[r])
        ctx_mask[i, :len(rows)] = ctx.astype(np.float32)
        tgt_mask[i, :len(rows)] = tm.astype(np.float32)
    t = lambda x: torch.as_tensor(x)
    return {"cont": t(cont), "act": t(act), "pri": t(pri), "sec": t(sec),
            "types": t(typ), "vid": t(vid),
            "ctx_mask": t(ctx_mask), "tgt_mask": t(tgt_mask)}
