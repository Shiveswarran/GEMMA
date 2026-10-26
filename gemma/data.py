"""Massive-STEPS loading and identifier-free check-in features.

A city's check-ins are read from ``<data_root>/<city>/<city>_checkins_<split>.csv``.  Every
check-in is described only by quantities that exist in every city:

    text embedding of "<category> : <name>"            (TextEncoder)
    Fourier features of hour of day, weekday, month    (temporal_features)
    log-distance, sin/cos bearing from the previous check-in of the trajectory, is-first flag
    activity / primary / leaf category                 (Taxonomy)

Place identifiers are kept only to know which check-ins refer to the same place (counts,
transitions, recency); they never enter a learned parameter.

For pretraining, all trajectories of a user are joined into one *diary* with a [SEP] token
between trajectories (``DiaryDataset``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .taxonomy import Taxonomy

TT_CHECKIN, TT_SEP = 0, 1                 # token types


def load_city_csv(root, city, splits) -> pd.DataFrame:
    frames = []
    for sp in ([splits] if isinstance(splits, str) else list(splits)):
        p = os.path.join(root, city, f"{city}_checkins_{sp}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["_split"] = sp
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no splits {splits} for {city} under {root}")
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp", "venue_id"]).reset_index(drop=True)


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    a_lat, a_lon, b_lat, b_lon = map(radians, [a_lat, a_lon, b_lat, b_lon])
    h = sin((b_lat - a_lat) / 2) ** 2 + cos(a_lat) * cos(b_lat) * sin((b_lon - a_lon) / 2) ** 2
    return 6371.0 * 2 * atan2(sqrt(h), sqrt(1 - h))


def bearing_rad(a_lat, a_lon, b_lat, b_lon):
    a_lat, b_lat = radians(a_lat), radians(b_lat)
    d_lon = radians(b_lon - a_lon)
    return atan2(sin(d_lon) * cos(b_lat), cos(a_lat) * sin(b_lat) - sin(a_lat) * cos(b_lat) * cos(d_lon))


def _cyclic(v, period, n_harm):
    out = []
    for k in range(1, n_harm + 1):
        out += [np.sin(2 * np.pi * k * v / period), np.cos(2 * np.pi * k * v / period)]
    return np.stack(out, -1).astype(np.float32)


def temporal_features(ts: pd.Series, n_harm: int) -> np.ndarray:
    tod = ts.dt.hour + ts.dt.minute / 60.0
    return np.concatenate([_cyclic(tod.values, 24, n_harm),
                           _cyclic(ts.dt.dayofweek.astype(float).values, 7, n_harm),
                           _cyclic(ts.dt.month.astype(float).values, 12, n_harm)], axis=1)


@dataclass
class CityData:
    city: str
    df: pd.DataFrame
    cont: np.ndarray          # (N, cont_dim) per check-in: [text | time | relative geometry]
    name_emb: np.ndarray      # (N, text_dim)
    venue_id: np.ndarray
    act_id: np.ndarray
    pri_id: np.ndarray
    sec_id: np.ndarray
    trails: List[dict]        # {user_id, trail_id, rows, split, t0}
    poi_table: Dict           # one row per place: text, taxonomy, coordinates
    codes: np.ndarray = None  # (N, sid_levels) semantic code of each check-in's place

    @property
    def cont_dim(self):
        return self.cont.shape[1]


def build_city_data(cfg, city, splits, tax: Taxonomy, text_enc, logger=None,
                    filter_users=True) -> CityData:
    df = load_city_csv(cfg.data_root, city, splits)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    coord_valid = (np.isfinite(lat) & np.isfinite(lon)).values
    if "venue_city_latitude" in df.columns:     # missing coordinates -> city centroid
        lat = lat.fillna(pd.to_numeric(df["venue_city_latitude"], errors="coerce"))
        lon = lon.fillna(pd.to_numeric(df["venue_city_longitude"], errors="coerce"))
    lat = lat.values.astype(float); lon = lon.values.astype(float)
    if logger and (~coord_valid).any():
        logger.info(f"[{city}] {int((~coord_valid).sum())}/{len(df)} rows without exact "
                    f"coordinates (city centroid used)")

    names = (df["venue_category"].astype(str) + " : " +
             df.get("name", pd.Series([""] * len(df))).astype(str)).tolist()
    name_emb = text_enc.encode(names)

    temporal = temporal_features(df["timestamp"], cfg.n_time_harmonics)
    spat = np.zeros((len(df), 4), np.float32)
    trail = df["trail_id"].values
    prev_i, prev_tr = -1, None
    for i in range(len(df)):
        if trail[i] != prev_tr:
            spat[i] = [0, 0, 0, 1]
        else:
            d = haversine_km(lat[prev_i], lon[prev_i], lat[i], lon[i])
            b = bearing_rad(lat[prev_i], lon[prev_i], lat[i], lon[i])
            d = 0.0 if not np.isfinite(d) else d
            b = 0.0 if not np.isfinite(b) else b
            spat[i] = [np.log1p(max(d, 0.0)), np.sin(b), np.cos(b), 0]
        prev_i, prev_tr = i, trail[i]
    cont = np.nan_to_num(np.concatenate([name_emb, temporal, spat], 1).astype(np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)

    cat_id = df["venue_category_id"].values
    act_id = np.array([tax.activity_id(c) for c in cat_id], np.int64)
    pri_id = np.array([tax.primary_id(c) for c in cat_id], np.int64)
    sec_id = np.array([tax.secondary_id(c) for c in cat_id], np.int64)

    trails = []
    for (uid, tid), g in df.groupby(["user_id", "trail_id"], sort=False):
        rows = g.index.to_numpy()
        if len(rows) < cfg.min_trail_len:
            continue
        trails.append({"user_id": int(uid), "trail_id": int(tid), "rows": rows,
                       "split": g["_split"].iloc[0], "t0": g["timestamp"].iloc[0]})
    if filter_users:
        by_user = {}
        for t in trails:
            by_user.setdefault(t["user_id"], []).append(t)
        trails = [t for ts in by_user.values() if len(ts) >= cfg.min_user_trails for t in ts]

    poi_table = _poi_table(df, name_emb, act_id, pri_id, sec_id, lat, lon, coord_valid)

    codes = None
    sid_path = os.path.join(cfg.sid_cache, f"{city}_codes.npz")
    if cfg.use_semantic_id and os.path.exists(sid_path):
        z = np.load(sid_path, allow_pickle=True)
        v2c = {int(v): c for v, c in zip(z["venue_ids"], z["codes"])}
        codes = np.zeros((len(df), z["codes"].shape[1]), np.int64)
        vids = df["venue_id"].values.astype(np.int64)
        for i in range(len(df)):
            c = v2c.get(int(vids[i]))
            if c is not None:
                codes[i] = c
        if logger:
            logger.info(f"[{city}] semantic codes ({codes.shape[1]} levels) for {len(v2c)} places")

    return CityData(city, df, cont, name_emb, df["venue_id"].values.astype(np.int64),
                    act_id, pri_id, sec_id, trails, poi_table, codes)


def _poi_table(df, name_emb, act, pri, sec, lat, lon, coord_valid):
    """One row per place (sorted venue ids): mean text embedding, majority category, mean
    coordinates."""
    vids = df["venue_id"].values.astype(np.int64)
    order = np.argsort(vids, kind="stable")
    uniq, start = np.unique(vids[order], return_index=True)
    end = np.r_[start[1:], len(order)]
    P = len(uniq)
    nm = np.zeros((P, name_emb.shape[1]), np.float32)
    a, p, s = (np.zeros(P, np.int64) for _ in range(3))
    plat, plon = np.zeros(P, np.float32), np.zeros(P, np.float32)
    cnt, valid = np.zeros(P, np.float32), np.ones(P, np.float32)
    la, lo = np.nan_to_num(lat).astype(np.float32), np.nan_to_num(lon).astype(np.float32)
    for k in range(P):
        sel = order[start[k]:end[k]]
        nm[k] = name_emb[sel].mean(0)
        a[k] = np.bincount(act[sel]).argmax()
        p[k] = np.bincount(pri[sel]).argmax()
        s[k] = np.bincount(sec[sel]).argmax()
        plat[k] = la[sel].mean(); plon[k] = lo[sel].mean(); cnt[k] = len(sel)
        valid[k] = float(coord_valid[sel].mean())
    return {"venue_ids": uniq, "name_emb": nm, "act": a, "pri": p, "sec": s, "lat": plat,
            "lon": plon, "count": cnt, "coord_valid": valid,
            "id2row": {int(v): i for i, v in enumerate(uniq)}}


# ----------------------------------------------------------------------------- pretraining
class DiaryDataset(torch.utils.data.Dataset):
    """One item = one user's diary (trajectories joined by [SEP], capped) with a target mask.

    Masking strategies:
      block_future : a block of future check-ins is hidden and predicted from the rest
      trail_end    : the last k check-ins of one trajectory are predicted from the strict past
                     (earlier trajectories + the trajectory's own prefix), i.e. the next-place
                     query itself
      mixed        : trail_end or block_future with probability 1/2 each (default)
    """

    def __init__(self, city: CityData, cfg):
        self.cfg = cfg
        by_user = {}
        for t in city.trails:
            by_user.setdefault(t["user_id"], []).append(t)
        self.diaries = []
        for ts in by_user.values():
            rows, types = [], []
            for j, t in enumerate(sorted(ts, key=lambda x: x["rows"][0])):
                if j > 0:
                    rows.append(-1); types.append(TT_SEP)
                for r in t["rows"]:
                    rows.append(int(r)); types.append(TT_CHECKIN)
                if len(rows) >= cfg.max_diary_len:
                    break
            rows, types = rows[:cfg.max_diary_len], types[:cfg.max_diary_len]
            if sum(1 for x in types if x == TT_CHECKIN) >= 3:
                self.diaries.append((np.array(rows), np.array(types)))

    def __len__(self):
        return len(self.diaries)

    def _block_future(self, n_ck, rng):
        m = np.zeros(n_ck, bool)
        b = min(self.cfg.mask_block_max, max(1, int(self.cfg.mask_ratio * n_ck)))
        cut = rng.randint(1, max(2, n_ck - b + 1))
        m[cut:cut + b] = True
        m[0] = False
        return m

    def _trail_end(self, types, rng):
        L = len(types)
        seps = [-1] + [i for i in range(L) if types[i] == TT_SEP] + [L]
        segs = [(seps[j] + 1, seps[j + 1]) for j in range(len(seps) - 1)
                if seps[j + 1] - (seps[j] + 1) >= 2]
        if not segs:
            return None
        a, b = segs[rng.randint(len(segs))]
        k = rng.randint(1, min(self.cfg.mask_block_max, b - a - 1) + 1)
        tmask = np.zeros(L, bool); tmask[b - k:b] = True
        ctx = np.zeros(L, bool); ctx[:b - k] = True
        return ctx, tmask

    def __getitem__(self, idx):
        rows, types = self.diaries[idx]
        rng = np.random.RandomState(int(torch.randint(0, 2 ** 31 - 1, (1,))))   # fresh mask
        strat = self.cfg.masking
        if strat == "trail_end" or (strat == "mixed" and rng.rand() < 0.5):
            out = self._trail_end(types, rng)
            if out is not None:
                return (rows, types) + out
        ck_pos = np.where(types == TT_CHECKIN)[0]
        tmask = np.zeros(len(rows), bool)
        tmask[ck_pos[self._block_future(len(ck_pos), rng)]] = True
        return rows, types, ~tmask, tmask


def collate(batch, city: CityData, cfg):
    B, L = len(batch), max(len(b[0]) for b in batch)
    cont = np.zeros((B, L, city.cont_dim), np.float32)
    act, pri, sec, typ = (np.zeros((B, L), np.int64) for _ in range(4))
    vid = np.full((B, L), -1, np.int64)
    Lc = city.codes.shape[1] if city.codes is not None else 0
    code = np.zeros((B, L, Lc), np.int64) if Lc else None
    ctx_mask, tgt_mask = np.zeros((B, L), np.float32), np.zeros((B, L), np.float32)
    for i, (rows, types, ctx, tm) in enumerate(batch):
        for j, r in enumerate(rows):
            typ[i, j] = types[j]
            if r >= 0:
                cont[i, j] = city.cont[r]
                act[i, j], pri[i, j], sec[i, j] = city.act_id[r], city.pri_id[r], city.sec_id[r]
                vid[i, j] = int(city.venue_id[r])
                if code is not None:
                    code[i, j] = city.codes[r]
        ctx_mask[i, :len(rows)] = ctx.astype(np.float32)
        tgt_mask[i, :len(rows)] = tm.astype(np.float32)
    t = torch.as_tensor
    out = {"cont": t(cont), "act": t(act), "pri": t(pri), "sec": t(sec), "types": t(typ),
           "vid": t(vid), "ctx_mask": t(ctx_mask), "tgt_mask": t(tgt_mask)}
    if code is not None:
        out["code"] = t(code)
    return out
