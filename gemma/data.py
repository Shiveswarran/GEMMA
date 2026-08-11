"""
gemma.data
----------
Loads Massive-STEPS check-in CSVs and turns them into everything the model
needs.  Key ideas that follow the paper and the user's design notes:

  * Trails are short (median ~2-3 check-ins), which is too short for a JEPA
    SSL signal on its own.  We therefore build a **per-user diary**: all of a
    user's trails, ordered in time, concatenated with a learned [SEP] token
    between trails (and a [USR] token at the front).  JEPA then predicts masked
    *future* check-ins across this longer diary, so the user embedding and the
    activity grammar get a real training signal.

  * The evaluation protocol matches the Massive-STEPS repo exactly: each trail's
    last check-in is the target; the context is every earlier check-in of that
    trail; predictions are ranked over the target city's POI vocabulary.

Public objects
--------------
CityData                 - per-city arrays + diaries + POI table (realization map)
build_category_vocab     - global category vocab shared across all cities
PretrainDiaryDataset     - torch Dataset yielding (context, target-block) windows
collate_windows          - pads/stacks a batch of windows
CONT_DIM / token type ids
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import (NameEmbedder, CategoryVocab, temporal_features,
                       temporal_dim, spatial_dim)
from .utils import haversine_km, bearing_rad

# token types
TT_CHECKIN, TT_SEP, TT_USR = 0, 1, 2

REQUIRED_COLS = ["trail_id", "user_id", "venue_id", "latitude", "longitude",
                 "name", "venue_category", "timestamp"]


# --------------------------------------------------------------------------- #
#  Loading helpers                                                            #
# --------------------------------------------------------------------------- #
def _split_path(data_root: str, city: str, split: str) -> str:
    return os.path.join(data_root, city, f"{city}_checkins_{split}.csv")


def load_city_csv(data_root: str, city: str, split: str) -> pd.DataFrame:
    path = _split_path(data_root, city, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "venue_id"])
    return df


def _filter_min(df: pd.DataFrame, min_trail_len: int,
                min_user_trails: int) -> pd.DataFrame:
    """Drop short trails and low-activity users (repo default: >=2 / >=3)."""
    tl = df.groupby("trail_id")["venue_id"].transform("size")
    df = df[tl >= min_trail_len]
    ut = df.groupby("user_id")["trail_id"].transform("nunique")
    df = df[ut >= min_user_trails]
    return df.reset_index(drop=True)


def build_category_vocab(data_root: str, cities: List[str],
                         splits=("train",)) -> CategoryVocab:
    """Fit ONE global category vocabulary over all cities (shared label space)."""
    vocab = CategoryVocab()
    cats: List[str] = []
    for city in cities:
        for split in splits:
            p = _split_path(data_root, city, split)
            if os.path.exists(p):
                cats.extend(pd.read_csv(p, usecols=["venue_category"])
                            ["venue_category"].astype(str).tolist())
    return vocab.fit(cats)


# --------------------------------------------------------------------------- #
#  Per-city container                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class CityData:
    city: str
    df: pd.DataFrame                      # sorted check-ins
    cont: np.ndarray                     # (N, CONT_DIM) continuous streams
    cat_id: np.ndarray                   # (N,) global category id
    primary_id: np.ndarray               # (N,) coarse primary category id
    venue_id: np.ndarray                 # (N,) city-local POI id
    user_ids: np.ndarray                 # (N,)
    diaries: List[Dict]                  # per-user ordered token plan
    poi_table: Dict                      # realization map inputs for the city
    trails: List[Dict]                   # per-trail row indices (for eval/pool)

    @property
    def cont_dim(self) -> int:
        return self.cont.shape[1]


def _row_temporal(ts: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hour = ts.dt.hour.values + ts.dt.minute.values / 60.0
    dow = ts.dt.dayofweek.values.astype(np.float32)
    month = (ts.dt.month.values - 1).astype(np.float32)
    return hour.astype(np.float32), dow, month


def build_city_data(data_root: str, city: str, split: str,
                    vocab: CategoryVocab, name_embedder: NameEmbedder,
                    cfg, logger=None, filter_users: bool = True) -> CityData:
    """Load one split of one city and precompute all city-agnostic features.

    filter_users=True  -> apply repo filters (trail>=2 AND user>=3 trails);
                          used for pretraining diaries.
    filter_users=False -> only drop trails with <2 check-ins; used for eval so
                          that every provided test trail is scored.
    """
    df = load_city_csv(data_root, city, split)
    min_ut = cfg.min_user_trails if filter_users else 1
    df = _filter_min(df, cfg.min_trail_len, min_ut)
    df = df.sort_values(["user_id", "trail_id", "timestamp"]).reset_index(drop=True)
    n = len(df)
    if logger:
        logger.info(f"[{city}/{split}] {n} check-ins, "
                    f"{df['user_id'].nunique()} users, "
                    f"{df['trail_id'].nunique()} trails")

    # --- name + category text embeddings (frozen, cached) ------------------
    name_emb = name_embedder.encode(df["name"].astype(str).tolist())
    cat_id = np.array([vocab.cat_id(c) for c in df["venue_category"]], dtype=np.int64)
    primary_id = np.array([vocab.primary_id(c) for c in df["venue_category"]],
                          dtype=np.int64)
    if cfg.cat_sbert:
        cat_emb = name_embedder.encode(df["venue_category"].astype(str).tolist())
    else:
        cat_emb = np.zeros((n, 0), dtype=np.float32)

    # --- temporal ----------------------------------------------------------
    hour, dow, month = _row_temporal(df["timestamp"])

    # --- relative spatial + dwell, computed within each (user,trail) -------
    dwell_log = np.zeros(n, dtype=np.float32)
    spat = np.zeros((n, spatial_dim()), dtype=np.float32)
    lat = df["latitude"].values.astype(float)
    lon = df["longitude"].values.astype(float)
    ts = df["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    trail = df["trail_id"].values
    prev_i = -1
    prev_trail = None
    for i in range(n):
        if trail[i] != prev_trail:                      # start of a trail
            spat[i] = [0.0, 0.0, 0.0, 1.0]              # is_first = 1
            dwell_log[i] = 0.0
        else:
            d = haversine_km(lat[prev_i], lon[prev_i], lat[i], lon[i])
            b = bearing_rad(lat[prev_i], lon[prev_i], lat[i], lon[i])
            spat[i] = [np.log1p(d), np.sin(b), np.cos(b), 0.0]
            dt = max(0.0, (ts[i] - ts[prev_i]) / 3600.0)   # hours
            dwell_log[i] = np.log1p(dt)
        prev_i = i
        prev_trail = trail[i]

    temporal = temporal_features(hour, dow, month, dwell_log, cfg.n_time_harmonics)

    cont = np.concatenate([name_emb, cat_emb, temporal, spat], axis=1).astype(np.float32)

    # --- per-user diaries (token plan) -------------------------------------
    diaries = _build_diaries(df, cfg.max_user_seq_len)

    # --- per-trail row lists (ordered) for eval / few-shot pools -----------
    trails = []
    for tid, g in df.groupby("trail_id", sort=False):
        rows = g.index.values.astype(np.int64)
        if len(rows) >= cfg.min_trail_len:
            trails.append({"trail_id": tid,
                           "user_id": g["user_id"].iloc[0],
                           "rows": rows})

    # --- POI realization table for the city --------------------------------
    poi_table = _build_poi_table(df, name_emb, cat_id, primary_id,
                                 hour, name_embedder, cfg)

    return CityData(city, df, cont, cat_id, primary_id,
                    df["venue_id"].values.astype(np.int64),
                    df["user_id"].values, diaries, poi_table, trails)


def _build_diaries(df: pd.DataFrame, max_len: int) -> List[Dict]:
    """Each diary is a token plan: an ordered list of (row_index, token_type)
    with [SEP] inserted between trails.  Row indices reference the city arrays.
    """
    diaries = []
    for uid, g in df.groupby("user_id", sort=False):
        idx = g.index.values
        trails = g["trail_id"].values
        plan_rows, plan_types = [], []
        prev_trail = None
        for j, ridx in enumerate(idx):
            if prev_trail is not None and trails[j] != prev_trail:
                plan_rows.append(-1)            # SEP marker
                plan_types.append(TT_SEP)
            plan_rows.append(int(ridx))
            plan_types.append(TT_CHECKIN)
            prev_trail = trails[j]
        # keep the most recent tokens if too long
        if len(plan_rows) > max_len:
            plan_rows = plan_rows[-max_len:]
            plan_types = plan_types[-max_len:]
        diaries.append({"user_id": uid,
                        "rows": np.array(plan_rows, dtype=np.int64),
                        "types": np.array(plan_types, dtype=np.int64)})
    return diaries


def _build_poi_table(df, name_emb, cat_id, primary_id, hour,
                     name_embedder, cfg) -> Dict:
    """Realization map R_c: for every POI in the city, a static feature vector
    (mean over its check-ins) so we can embed candidate POIs into activity space
    at transfer time, plus a visit-count popularity prior."""
    vids = df["venue_id"].values
    order = np.argsort(vids, kind="stable")
    uniq, start = np.unique(vids[order], return_index=True)
    end = np.append(start[1:], len(vids))
    name = np.zeros((len(uniq), name_emb.shape[1]), dtype=np.float32)
    catv = np.zeros(len(uniq), dtype=np.int64)
    primv = np.zeros(len(uniq), dtype=np.int64)
    meanhour = np.zeros(len(uniq), dtype=np.float32)
    count = np.zeros(len(uniq), dtype=np.float32)
    for k in range(len(uniq)):
        sel = order[start[k]:end[k]]
        name[k] = name_emb[sel].mean(0)
        catv[k] = np.bincount(cat_id[sel]).argmax()
        primv[k] = np.bincount(primary_id[sel]).argmax()
        meanhour[k] = hour[sel].mean()
        count[k] = len(sel)
    if cfg.cat_sbert:
        cat_strings = df.drop_duplicates("venue_id").set_index("venue_id")\
            .loc[uniq, "venue_category"].astype(str).tolist()
        cat_text = name_embedder.encode(cat_strings)
    else:
        cat_text = np.zeros((len(uniq), 0), dtype=np.float32)
    return {"venue_ids": uniq, "name_emb": name, "cat_text": cat_text,
            "cat_id": catv, "primary_id": primv, "mean_hour": meanhour,
            "count": count, "id2row": {int(v): i for i, v in enumerate(uniq)}}


# --------------------------------------------------------------------------- #
#  JEPA windowing dataset                                                     #
# --------------------------------------------------------------------------- #
class PretrainDiaryDataset:
    """Yields (context, target-block) windows sampled from user diaries.

    For a diary, a split point is drawn so that there is >=1 context check-in
    and a contiguous block of up to `mask_block_max` future check-ins.  The
    context tokens are handed to the context encoder; the target check-ins are
    the JEPA prediction targets (their latents + activity assignments).
    """

    def __init__(self, cities: List[CityData], cfg, rng_seed: int = 0):
        self.cont = [c.cont for c in cities]
        self.cat_id = [c.cat_id for c in cities]
        self.primary_id = [c.primary_id for c in cities]
        self.cfg = cfg
        # flatten diaries, remembering which city they belong to
        self.index: List[Tuple[int, int]] = []
        self.diaries = []
        for ci, c in enumerate(cities):
            for d in c.diaries:
                # need at least 2 check-ins to form context+target
                n_checkin = int((d["types"] == TT_CHECKIN).sum())
                if n_checkin >= 2:
                    self.index.append((ci, len(self.diaries)))
                    self.diaries.append(d)
        self.rng = np.random.RandomState(rng_seed)
        self.cont_dim = cities[0].cont_dim

    def __len__(self):
        return len(self.index)

    def _gather(self, ci, rows, types):
        """Materialise continuous + id features for a token plan slice."""
        L = len(rows)
        cont = np.zeros((L, self.cont_dim), dtype=np.float32)
        cid = np.zeros(L, dtype=np.int64)
        pid = np.zeros(L, dtype=np.int64)
        for k, (r, t) in enumerate(zip(rows, types)):
            if t == TT_CHECKIN and r >= 0:
                cont[k] = self.cont[ci][r]
                cid[k] = self.cat_id[ci][r]
                pid[k] = self.primary_id[ci][r]
        return cont, cid, pid

    def __getitem__(self, i):
        ci, di = self.index[i]
        d = self.diaries[di]
        rows, types = d["rows"], d["types"]
        checkin_pos = np.where(types == TT_CHECKIN)[0]
        # split so target block is the last few check-ins of a random window
        n_ck = len(checkin_pos)
        # choose how many future check-ins to predict
        blk = self.rng.randint(self.cfg.mask_block_min,
                               min(self.cfg.mask_block_max, n_ck - 1) + 1)
        split_ck = self.rng.randint(1, n_ck - blk + 1)     # >=1 context check-in
        ctx_end_pos = checkin_pos[split_ck - 1] + 1        # token index (exclusive)
        tgt_positions = checkin_pos[split_ck:split_ck + blk]

        ctx_rows = rows[:ctx_end_pos]
        ctx_types = types[:ctx_end_pos]
        # keep the last max_context_len tokens
        if len(ctx_rows) > self.cfg.max_context_len:
            ctx_rows = ctx_rows[-self.cfg.max_context_len:]
            ctx_types = ctx_types[-self.cfg.max_context_len:]

        ctx_cont, ctx_cid, ctx_pid = self._gather(ci, ctx_rows, ctx_types)
        tgt_cont, tgt_cid, tgt_pid = self._gather(
            ci, rows[tgt_positions], types[tgt_positions])
        # relative offset of each target from the context end (for the predictor)
        offsets = (tgt_positions - ctx_end_pos + 1).astype(np.int64)
        return {
            "ctx_cont": ctx_cont, "ctx_cid": ctx_cid, "ctx_pid": ctx_pid,
            "ctx_types": ctx_types.astype(np.int64),
            "tgt_cont": tgt_cont, "tgt_cid": tgt_cid, "tgt_pid": tgt_pid,
            "offsets": offsets,
        }


def collate_windows(batch, block_max: int):
    """Pad context to the batch max length; pad target block to block_max."""
    B = len(batch)
    ctx_lens = [len(b["ctx_types"]) for b in batch]
    Lc = max(ctx_lens)
    cont_dim = batch[0]["ctx_cont"].shape[1]

    ctx_cont = np.zeros((B, Lc, cont_dim), dtype=np.float32)
    ctx_cid = np.zeros((B, Lc), dtype=np.int64)
    ctx_pid = np.zeros((B, Lc), dtype=np.int64)
    ctx_types = np.full((B, Lc), TT_CHECKIN, dtype=np.int64)
    ctx_mask = np.zeros((B, Lc), dtype=np.float32)          # 1 = real token

    tgt_cont = np.zeros((B, block_max, cont_dim), dtype=np.float32)
    tgt_cid = np.zeros((B, block_max), dtype=np.int64)
    tgt_pid = np.zeros((B, block_max), dtype=np.int64)
    tgt_mask = np.zeros((B, block_max), dtype=np.float32)
    offsets = np.zeros((B, block_max), dtype=np.int64)

    for i, b in enumerate(batch):
        lc = len(b["ctx_types"])
        ctx_cont[i, :lc] = b["ctx_cont"]
        ctx_cid[i, :lc] = b["ctx_cid"]
        ctx_pid[i, :lc] = b["ctx_pid"]
        ctx_types[i, :lc] = b["ctx_types"]
        ctx_mask[i, :lc] = 1.0
        bt = len(b["offsets"])
        tgt_cont[i, :bt] = b["tgt_cont"]
        tgt_cid[i, :bt] = b["tgt_cid"]
        tgt_pid[i, :bt] = b["tgt_pid"]
        tgt_mask[i, :bt] = 1.0
        offsets[i, :bt] = b["offsets"]

    import torch
    t = torch.as_tensor
    return {
        "ctx_cont": t(ctx_cont), "ctx_cid": t(ctx_cid), "ctx_pid": t(ctx_pid),
        "ctx_types": t(ctx_types), "ctx_mask": t(ctx_mask),
        "tgt_cont": t(tgt_cont), "tgt_cid": t(tgt_cid), "tgt_pid": t(tgt_pid),
        "tgt_mask": t(tgt_mask), "offsets": t(offsets),
    }
