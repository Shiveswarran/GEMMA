#!/usr/bin/env python3
"""
JEPA SSL User Tower (per-user SSL with session separators) + per-trail finetuning for Next-POI.

Task: for each trail_id, target = last POI, input = prefix (all POIs before last).

PROTOCOL:
1) Build vocab from TRAIN ONLY (default, leakage-safe). Unseen tokens in valid/test -> UNK.
   (Optional: vocab_from_all_splits=True if benchmark assumes closed-world across splits.)
2) Build per-user TRAIN histories by concatenating that user’s TRAIN trails in time order
   and insert <SEP> between trails.
3) JEPA SSL pretraining on user histories (TRAIN only).
4) Supervised finetuning on TRAIN trails (prefix->last POI).
5) Evaluate on VALID and TEST trails (no training).

LOGGING/SAVING (run_dir):
- run.log
- config.json
- data_stats.json (trail length stats, dropped samples)
- train_ssl_log.csv
- train_ft_log.csv
- valid_log.csv
- test_metrics.json
- checkpoints/
    ckpt_ssl_last.pt
    ckpt_ft_best.pt
    ckpt_ft_last.pt
"""

import os
import math
import json
import time
import random
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Repro
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Run dir + logging + save utilities
# ============================================================
def make_run_dir(out_dir: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"run_{ts}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return str(run_dir)

def setup_logger(run_dir: str) -> logging.Logger:
    logger = logging.getLogger("JEPA_RUN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), mode="w")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_csv_row(path: str, header: List[str], row: Dict):
    exists = os.path.exists(path)
    with open(path, "a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(h, "")) for h in header) + "\n")

def save_checkpoint(path: str, model: nn.Module, head: Optional[nn.Module], args_dict: Dict, extra: Dict):
    payload = {"model": model.state_dict(), "args": args_dict, "extra": extra}
    if head is not None:
        payload["head"] = head.state_dict()
    torch.save(payload, path)


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def acc_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=k, dim=-1).indices
    return (topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()

@torch.no_grad()
def ndcg_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=k, dim=-1).indices
    matches = (topk == labels.unsqueeze(-1))
    ranks = torch.argmax(matches.int(), dim=-1)  # 0..k-1
    has = matches.any(dim=-1)
    dcg = torch.zeros_like(labels, dtype=torch.float)
    dcg[has] = 1.0 / torch.log2((ranks[has].float() + 2.0))
    return dcg.mean().item()


# ============================================================
# Vocab with special tokens (PAD, UNK, SEP)
# ============================================================
class Vocab:
    """
    Token -> id mapping with predefined specials.
    """
    def __init__(self, specials: List[str]):
        assert len(specials) >= 2, "Need at least PAD and UNK."
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        for sp in specials:
            self._add_internal(sp)

    def _add_internal(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def add(self, token) -> int:
        return self._add_internal(str(token))

    def get(self, token) -> int:
        return self.stoi.get(str(token), self.stoi["<UNK>"])

    def __len__(self):
        return len(self.itos)


# ============================================================
# Data loading / parsing
# ============================================================
REQUIRED_COLS = ["trail_id", "user_id", "venue_id", "latitude", "longitude", "timestamp"]

def load_checkins(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    if "venue_category_id_code" not in df.columns:
        df["venue_category_id_code"] = 0
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["trail_id", "user_id", "venue_id", "timestamp"])
    return df

def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=0) -> int:
    try:
        if pd.isna(x): return default
        return int(float(x))
    except Exception:
        return default

def time_delta_bucket(delta_min: float) -> int:
    if delta_min <= 5: return 0
    if delta_min <= 15: return 1
    if delta_min <= 30: return 2
    if delta_min <= 60: return 3
    if delta_min <= 120: return 4
    if delta_min <= 240: return 5
    if delta_min <= 480: return 6
    if delta_min <= 1440: return 7
    if delta_min <= 4320: return 8
    return 9

def trail_length_stats(trails: Dict[str, Dict]) -> Dict[str, float]:
    lens = [len(t["venue"]) for t in trails.values()]
    if not lens:
        return {"count": 0}
    arr = np.array(lens)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


# ============================================================
# Build trails (per trail_id) with leakage-safe vocab handling
# ============================================================
def build_trails(
    df: pd.DataFrame,
    venue_vocab: Vocab,
    user_vocab: Vocab,
    cat_vocab: Vocab,
    add_new_tokens: bool,
) -> Dict[str, Dict]:
    """
    trails[trail_id] = {
        "user": int,
        "venue": [..],
        "cat": [..],
        "lat":[..],
        "lon":[..],
        "hour":[..],
        "dow":[..],
        "dtb":[..]
    }
    If add_new_tokens=False: use vocab.get() (UNK for unseen).
    """
    df = df.sort_values(["trail_id", "timestamp"], ascending=True)
    trails: Dict[str, Dict] = {}

    for tid, g in df.groupby("trail_id", sort=False):
        g = g.sort_values("timestamp")
        user_id = g["user_id"].iloc[0]
        u = user_vocab.add(user_id) if add_new_tokens else user_vocab.get(user_id)

        venues, cats, lats, lons, hours, dows, dtb = [], [], [], [], [], [], []
        prev_ts = None
        for _, row in g.iterrows():
            vid = row["venue_id"]
            cid = _safe_int(row.get("venue_category_id_code", 0), 0)

            v = venue_vocab.add(vid) if add_new_tokens else venue_vocab.get(vid)
            c = cat_vocab.add(cid) if add_new_tokens else cat_vocab.get(cid)

            ts = row["timestamp"]
            lat = _safe_float(row.get("latitude", 0.0), 0.0)
            lon = _safe_float(row.get("longitude", 0.0), 0.0)
            hour = int(ts.hour)
            dow = int(ts.dayofweek)

            if prev_ts is None:
                delta_min = 0.0
            else:
                delta_min = max(0.0, (ts - prev_ts).total_seconds() / 60.0)
            dt_bucket = time_delta_bucket(delta_min)

            venues.append(v)
            cats.append(c)
            lats.append(lat)
            lons.append(lon)
            hours.append(hour)
            dows.append(dow)
            dtb.append(dt_bucket)

            prev_ts = ts

        if len(venues) < 2:
            continue

        trails[str(tid)] = {
            "trail_id": str(tid),
            "user": u,
            "venue": venues,
            "cat": cats,
            "lat": lats,
            "lon": lons,
            "hour": hours,
            "dow": dows,
            "dtb": dtb,
            # store start time for ordering if needed
            "start_ts": g["timestamp"].iloc[0].to_pydatetime(),
        }
    return trails


# ============================================================
# Build per-user TRAIN histories with session separators
# ============================================================
def build_user_histories_from_train_trails(
    train_trails: Dict[str, Dict],
    sep_venue_id: int,
    sep_cat_id: int,
) -> Dict[int, Dict]:
    """
    Creates per-user sequences by ordering that user's trails by start_ts and concatenating,
    inserting a <SEP> token between trails.

    Returns:
      user_hist[user_id_idx] = {
        "user": user_id_idx,
        "venue": [..],
        "cat": [..],
        "lat":[..],
        "lon":[..],
        "hour":[..],
        "dow":[..],
        "dtb":[..],
      }

    Note:
    - Separator has:
        venue=SEP, cat=SEP_CAT, lat/lon=0, hour=0, dow=0, dtb=0
    - dtb is reset per trail naturally because dtb in trails already computed with prev_ts inside trail.
      The SEP token itself also has dtb=0.
    """
    by_user: Dict[int, List[Dict]] = {}
    for tr in train_trails.values():
        by_user.setdefault(tr["user"], []).append(tr)

    user_hist: Dict[int, Dict] = {}
    for u, trs in by_user.items():
        trs = sorted(trs, key=lambda x: x["start_ts"])

        venues, cats, lats, lons, hours, dows, dtb = [], [], [], [], [], [], []
        first = True
        for tr in trs:
            if not first:
                # insert separator
                venues.append(sep_venue_id)
                cats.append(sep_cat_id)
                lats.append(0.0)
                lons.append(0.0)
                hours.append(0)
                dows.append(0)
                dtb.append(0)
            first = False

            venues.extend(tr["venue"])
            cats.extend(tr["cat"])
            lats.extend(tr["lat"])
            lons.extend(tr["lon"])
            hours.extend(tr["hour"])
            dows.extend(tr["dow"])
            dtb.extend(tr["dtb"])

        if len(venues) < 2:
            continue

        user_hist[u] = {
            "user": u,
            "venue": venues,
            "cat": cats,
            "lat": lats,
            "lon": lons,
            "hour": hours,
            "dow": dows,
            "dtb": dtb,
        }
    return user_hist


# ============================================================
# Tensor slicing + padding
# ============================================================
def slice_sequence(seq: Dict, start: int, end: int) -> Dict[str, torch.Tensor]:
    return {
        "user": torch.tensor(seq["user"], dtype=torch.long),
        "venue": torch.tensor(seq["venue"][start:end], dtype=torch.long),
        "cat": torch.tensor(seq["cat"][start:end], dtype=torch.long),
        "latlon": torch.tensor(np.stack([seq["lat"][start:end], seq["lon"][start:end]], axis=-1), dtype=torch.float),
        "hour": torch.tensor(seq["hour"][start:end], dtype=torch.long),
        "dow": torch.tensor(seq["dow"][start:end], dtype=torch.long),
        "dtb": torch.tensor(seq["dtb"][start:end], dtype=torch.long),
    }

def pad_1d(x: torch.Tensor, max_len: int, pad_val: int = 0) -> torch.Tensor:
    T = x.size(0)
    if T >= max_len: return x[:max_len]
    pad = torch.full((max_len - T,), pad_val, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def pad_2d(x: torch.Tensor, max_len: int, pad_val: float = 0.0) -> torch.Tensor:
    T = x.size(0)
    if T >= max_len: return x[:max_len]
    pad = torch.full((max_len - T, x.size(1)), pad_val, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


# ============================================================
# SSL dataset (per-user histories)
# ============================================================
class UserHistorySSLDataset(Dataset):
    """
    Samples windows from per-user histories.
    """
    def __init__(
        self,
        user_hist: Dict[int, Dict],
        max_len: int,
        min_len: int,
        use_full_prob: float,
        seed: int = 0,
    ):
        self.items = list(user_hist.values())
        self.max_len = max_len
        self.min_len = min_len
        self.use_full_prob = use_full_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        seq = self.items[idx]
        L = len(seq["venue"])

        if L <= self.min_len:
            return slice_sequence(seq, 0, L)

        if self.rng.random() < self.use_full_prob:
            start, end = 0, min(L, self.max_len)
            return slice_sequence(seq, start, end)

        hi = min(self.max_len, L)
        lo = min(self.min_len, hi)
        wlen = self.rng.randint(lo, hi)
        start = self.rng.randint(0, L - wlen)
        end = start + wlen
        return slice_sequence(seq, start, end)

def ssl_collate(batch: List[Dict], max_len: int):
    users = torch.stack([b["user"] for b in batch], dim=0)  # (B,)
    lens = torch.tensor([b["venue"].size(0) for b in batch], dtype=torch.long)

    venue = torch.stack([pad_1d(b["venue"], max_len, 0) for b in batch], dim=0)  # (B,T)
    cat = torch.stack([pad_1d(b["cat"], max_len, 0) for b in batch], dim=0)
    hour = torch.stack([pad_1d(b["hour"], max_len, 0) for b in batch], dim=0)
    dow = torch.stack([pad_1d(b["dow"], max_len, 0) for b in batch], dim=0)
    dtb = torch.stack([pad_1d(b["dtb"], max_len, 0) for b in batch], dim=0)
    latlon = torch.stack([pad_2d(b["latlon"], max_len, 0.0) for b in batch], dim=0)

    pad_mask = torch.arange(max_len).unsqueeze(0) >= lens.unsqueeze(1)
    return {"user": users, "venue": venue, "cat": cat, "hour": hour, "dow": dow,
            "dtb": dtb, "latlon": latlon, "pad_mask": pad_mask, "lens": lens}


# ============================================================
# Supervised dataset (per-trail prefix -> last)
# ============================================================
class NextPOIDataset(Dataset):
    def __init__(self, trails: Dict[str, Dict], max_len: int, unk_venue_id: int, drop_oov_label: bool):
        self.items = list(trails.values())
        self.max_len = max_len
        self.unk_venue_id = unk_venue_id
        self.drop_oov_label = drop_oov_label

        # pre-filter if requested
        if drop_oov_label:
            self.items = [t for t in self.items if t["venue"][-1] != unk_venue_id]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        tr = self.items[idx]
        L = len(tr["venue"])
        label = tr["venue"][L - 1]

        start = max(0, (L - 1) - self.max_len)
        end = L - 1
        x = slice_sequence(tr, start, end)
        x["label"] = torch.tensor(label, dtype=torch.long)
        return x

def sup_collate(batch: List[Dict], max_len: int):
    users = torch.stack([b["user"] for b in batch], dim=0)
    lens = torch.tensor([b["venue"].size(0) for b in batch], dtype=torch.long)

    venue = torch.stack([pad_1d(b["venue"], max_len, 0) for b in batch], dim=0)
    cat = torch.stack([pad_1d(b["cat"], max_len, 0) for b in batch], dim=0)
    hour = torch.stack([pad_1d(b["hour"], max_len, 0) for b in batch], dim=0)
    dow = torch.stack([pad_1d(b["dow"], max_len, 0) for b in batch], dim=0)
    dtb = torch.stack([pad_1d(b["dtb"], max_len, 0) for b in batch], dim=0)
    latlon = torch.stack([pad_2d(b["latlon"], max_len, 0.0) for b in batch], dim=0)

    pad_mask = torch.arange(max_len).unsqueeze(0) >= lens.unsqueeze(1)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"user": users, "venue": venue, "cat": cat, "hour": hour, "dow": dow,
            "dtb": dtb, "latlon": latlon, "pad_mask": pad_mask, "lens": lens, "label": labels}


# ============================================================
# JEPA Model
# ============================================================
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]

class UserSequenceEncoder(nn.Module):
    """
    Token representation includes:
      venue + cat + hour + dow + dtb + geo + user_embedding
    Works for both SSL (user-history windows) and finetune (trail prefixes).
    """
    def __init__(
        self,
        num_venues: int,
        num_users: int,
        num_cats: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_len: int,
        dtb_bins: int = 10,
    ):
        super().__init__()
        self.d_model = d_model

        self.venue_emb = nn.Embedding(num_venues, d_model)
        self.user_emb = nn.Embedding(num_users, d_model)
        self.cat_emb = nn.Embedding(num_cats, d_model)
        self.hour_emb = nn.Embedding(24, d_model)
        self.dow_emb = nn.Embedding(7, d_model)
        self.dtb_emb = nn.Embedding(dtb_bins, d_model)

        self.geo_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, user, venue, cat, hour, dow, dtb, latlon, pad_mask):
        B, T = venue.shape
        v = self.venue_emb(venue)
        c = self.cat_emb(cat)
        h = self.hour_emb(hour)
        d = self.dow_emb(dow)
        g = self.dtb_emb(dtb)
        geo = self.geo_mlp(latlon)
        u = self.user_emb(user).unsqueeze(1).expand(B, T, self.d_model)

        x = v + c + h + d + g + geo + u
        x = self.fuse(x)
        x = self.pos(x)
        x = self.tr(x, src_key_padding_mask=pad_mask)
        return self.ln(x)

class PredictorTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        self.proj = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, ctx_latents: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        B, T, D = ctx_latents.shape
        m = mask.unsqueeze(-1)
        mask_tok = self.mask_token.expand(B, T, D)
        x = torch.where(m, mask_tok, ctx_latents)
        x = self.pos(x)
        x = self.tr(x)
        x = self.ln(x)
        return self.proj(x)

def vicreg_variance_loss(z: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()

def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    N, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, (N - 1))
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / D

@dataclass
class JEPALossConfig:
    mask_mse_weight: float = 1.0
    var_weight: float = 1.0
    cov_weight: float = 0.04
    target_std: float = 1.0

class JEPAUserTower(nn.Module):
    def __init__(self, num_venues, num_users, num_cats, d_model, enc_layers, pred_layers, nhead, dropout, max_len):
        super().__init__()
        self.context_encoder = UserSequenceEncoder(num_venues, num_users, num_cats, d_model, nhead, enc_layers, dropout, max_len)
        self.target_encoder = UserSequenceEncoder(num_venues, num_users, num_cats, d_model, nhead, enc_layers, dropout, max_len)
        self.predictor = PredictorTransformer(d_model, nhead, pred_layers, dropout, max_len)

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.d_model = d_model

    @torch.no_grad()
    def ema_update_target(self, momentum: float):
        for pt, ps in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def forward_ssl(self, batch: Dict[str, torch.Tensor], mask: torch.BoolTensor, loss_cfg: JEPALossConfig):
        user = batch["user"]; venue=batch["venue"]; cat=batch["cat"]; hour=batch["hour"]; dow=batch["dow"]
        dtb=batch["dtb"]; latlon=batch["latlon"]; pad_mask=batch["pad_mask"]

        ctx = self.context_encoder(user, venue, cat, hour, dow, dtb, latlon, pad_mask)
        with torch.no_grad():
            tgt = self.target_encoder(user, venue, cat, hour, dow, dtb, latlon, pad_mask)

        pred = self.predictor(ctx, mask)

        B, T, D = pred.shape
        flat_mask = mask.view(-1)
        pred_flat = pred.view(-1, D)[flat_mask]
        tgt_flat  = tgt.view(-1, D)[flat_mask]

        if pred_flat.numel() == 0:
            z = torch.tensor(0.0, device=pred.device)
            return z, {"mse": z, "var": z, "cov": z}

        mse = F.mse_loss(pred_flat, tgt_flat)
        var = vicreg_variance_loss(pred_flat, target_std=loss_cfg.target_std)
        cov = vicreg_covariance_loss(pred_flat)
        loss = loss_cfg.mask_mse_weight*mse + loss_cfg.var_weight*var + loss_cfg.cov_weight*cov
        return loss, {"mse": mse.detach(), "var": var.detach(), "cov": cov.detach()}

    def pooled_user_embedding(self, batch: Dict[str, torch.Tensor], pool: str = "mean"):
        user = batch["user"]; venue=batch["venue"]; cat=batch["cat"]; hour=batch["hour"]; dow=batch["dow"]
        dtb=batch["dtb"]; latlon=batch["latlon"]; pad_mask=batch["pad_mask"]

        z = self.context_encoder(user, venue, cat, hour, dow, dtb, latlon, pad_mask)  # (B,T,D)

        if pool == "mean":
            keep = (~pad_mask).unsqueeze(-1).float()
            denom = keep.sum(dim=1).clamp(min=1.0)
            return (z * keep).sum(dim=1) / denom
        elif pool == "last":
            lens = batch["lens"]
            idx = (lens - 1).clamp(min=0)
            return z[torch.arange(z.size(0), device=z.device), idx, :]
        else:
            raise ValueError("pool must be mean or last")

class NextPOIHead(nn.Module):
    def __init__(self, d_model: int, num_venues: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
        )
        self.out = nn.Linear(d_model, num_venues)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(u))


# ============================================================
# Masking for SSL (span mask + avoid PAD and SEP)
# ============================================================
def make_span_mask(seq_len: int, mask_ratio: float, mean_span_len: int, rng: random.Random) -> torch.BoolTensor:
    n_to_mask = max(1, int(round(seq_len * mask_ratio)))
    mask = np.zeros(seq_len, dtype=np.bool_)
    masked = 0
    guard = 0
    while masked < n_to_mask and guard < seq_len * 10:
        span = max(1, int(round(rng.expovariate(1.0 / max(1, mean_span_len)))))
        start = rng.randint(0, seq_len - 1)
        end = min(seq_len, start + span)
        for i in range(start, end):
            if not mask[i]:
                mask[i] = True
                masked += 1
                if masked >= n_to_mask:
                    break
        guard += 1
    return torch.tensor(mask, dtype=torch.bool)

def build_ssl_mask(pad_mask: torch.BoolTensor, venue_tokens: torch.Tensor, sep_id: int,
                   mask_ratio: float, mean_span_len: int, rng: random.Random) -> torch.BoolTensor:
    """
    Mask only real (non-pad) AND non-SEP tokens.
    """
    B, T = pad_mask.shape
    masks = []
    for b in range(B):
        L = int((~pad_mask[b]).sum().item())
        if L <= 1:
            masks.append(torch.zeros(T, dtype=torch.bool))
            continue
        base = make_span_mask(L, mask_ratio, mean_span_len, rng)  # (L,)
        full = torch.zeros(T, dtype=torch.bool)
        full[:L] = base

        # do not mask SEP tokens
        sep_positions = (venue_tokens[b] == sep_id)
        full = full & (~sep_positions)

        masks.append(full)
    return torch.stack(masks, dim=0)


# ============================================================
# Training loops (with logging + checkpoints)
# ============================================================
def pretrain_ssl(
    model: JEPAUserTower,
    loader: DataLoader,
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
    mask_ratio: float,
    mean_span_len: int,
    ema_momentum: float,
    loss_cfg: JEPALossConfig,
    grad_clip: float,
    log_every: int,
    save_every: int,
    run_dir: str,
    logger: logging.Logger,
    sep_venue_id: int,
    args_dict: Dict,
):
    model.to(device)
    model.train()

    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    ssl_csv = os.path.join(run_dir, "train_ssl_log.csv")
    ssl_header = ["step", "loss", "mse", "var", "cov", "lr", "time_sec"]

    rng = random.Random(0)
    it = iter(loader)
    t_step0 = time.time()

    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}
        mask = build_ssl_mask(batch["pad_mask"], batch["venue"], sep_venue_id, mask_ratio, mean_span_len, rng).to(device)

        loss, m = model.forward_ssl(batch, mask, loss_cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        with torch.no_grad():
            model.ema_update_target(momentum=ema_momentum)

        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t_step0

        append_csv_row(ssl_csv, ssl_header, {
            "step": step, "loss": float(loss.item()),
            "mse": float(m["mse"].item()), "var": float(m["var"].item()),
            "cov": float(m["cov"].item()), "lr": lr_now, "time_sec": dt
        })

        if step % log_every == 0:
            logger.info(
                f"[SSL] step={step}/{steps} loss={loss.item():.4f} mse={m['mse'].item():.4f} "
                f"var={m['var'].item():.4f} cov={m['cov'].item():.4f} lr={lr_now:.2e} dt={dt:.2f}s"
            )

        if save_every > 0 and step % save_every == 0:
            ckpt = os.path.join(run_dir, "checkpoints", f"ckpt_ssl_step{step}.pt")
            save_checkpoint(ckpt, model, None, args_dict, {"step": step})
            logger.info(f"Saved SSL checkpoint: {ckpt}")

        t_step0 = time.time()

    ckpt_last = os.path.join(run_dir, "checkpoints", "ckpt_ssl_last.pt")
    save_checkpoint(ckpt_last, model, None, args_dict, {"step": steps})
    logger.info(f"Saved SSL last checkpoint: {ckpt_last}")


@torch.no_grad()
def evaluate_next_poi(model: JEPAUserTower, head: NextPOIHead, loader: DataLoader, device: str, pool: str) -> Dict[str, float]:
    model.eval(); head.eval()
    a1=a5=a10=n5=n10=0.0
    n=0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["label"]
        u = model.pooled_user_embedding(batch, pool=pool)
        logits = head(u)

        bsz = labels.size(0)
        a1 += acc_at_k(logits, labels, 1) * bsz
        a5 += acc_at_k(logits, labels, 5) * bsz
        a10 += acc_at_k(logits, labels, 10) * bsz
        n5 += ndcg_at_k(logits, labels, 5) * bsz
        n10 += ndcg_at_k(logits, labels, 10) * bsz
        n += bsz
    n = max(1, n)
    return {"acc@1": a1/n, "acc@5": a5/n, "acc@10": a10/n, "ndcg@5": n5/n, "ndcg@10": n10/n}


def finetune_next_poi(
    model: JEPAUserTower,
    head: NextPOIHead,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    freeze_encoder: bool,
    pool: str,
    log_every: int,
    run_dir: str,
    logger: logging.Logger,
    best_metric: str,
    args_dict: Dict,
):
    model.to(device); head.to(device)

    if freeze_encoder:
        for p in model.context_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.context_encoder.parameters():
            p.requires_grad = True

    params = list(head.parameters()) + [p for p in model.context_encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    train_csv = os.path.join(run_dir, "train_ft_log.csv")
    train_header = ["epoch", "iter", "loss", "acc@1", "acc@5", "acc@10", "ndcg@5", "ndcg@10", "lr", "time_sec"]
    valid_csv = os.path.join(run_dir, "valid_log.csv")
    valid_header = ["epoch", "acc@1", "acc@5", "acc@10", "ndcg@5", "ndcg@10"]

    best_score = -1.0
    best_epoch = 0

    for ep in range(1, epochs+1):
        model.train(); head.train()
        t0 = time.time()
        running = 0.0

        for it, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]

            u = model.pooled_user_embedding(batch, pool=pool)
            logits = head(u)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            if it % log_every == 0:
                lr_now = opt.param_groups[0]["lr"]
                dt = time.time() - t0
                avg = running / log_every
                running = 0.0

                a1 = acc_at_k(logits.detach(), labels.detach(), 1)
                a5 = acc_at_k(logits.detach(), labels.detach(), 5)
                a10 = acc_at_k(logits.detach(), labels.detach(), 10)
                n5 = ndcg_at_k(logits.detach(), labels.detach(), 5)
                n10 = ndcg_at_k(logits.detach(), labels.detach(), 10)

                logger.info(f"[FT] ep={ep} it={it} loss={avg:.4f} acc1={a1:.3f} acc5={a5:.3f} acc10={a10:.3f} ndcg5={n5:.3f} ndcg10={n10:.3f}")
                append_csv_row(train_csv, train_header, {
                    "epoch": ep, "iter": it, "loss": avg,
                    "acc@1": a1, "acc@5": a5, "acc@10": a10,
                    "ndcg@5": n5, "ndcg@10": n10, "lr": lr_now, "time_sec": dt
                })
                t0 = time.time()

        # validation
        metrics = evaluate_next_poi(model, head, valid_loader, device, pool)
        append_csv_row(valid_csv, valid_header, {"epoch": ep, **metrics})
        logger.info(f"[VAL] ep={ep} metrics={metrics}")

        score = metrics.get(best_metric, metrics["acc@10"])
        if score > best_score:
            best_score = score
            best_epoch = ep
            ckpt_best = os.path.join(run_dir, "checkpoints", "ckpt_ft_best.pt")
            save_checkpoint(ckpt_best, model, head, args_dict, {"epoch": ep, "best_metric": best_metric, "best_score": best_score})
            logger.info(f"Saved BEST finetune checkpoint: {ckpt_best} ({best_metric}={best_score:.4f})")

    ckpt_last = os.path.join(run_dir, "checkpoints", "ckpt_ft_last.pt")
    save_checkpoint(ckpt_last, model, head, args_dict, {"epoch": epochs})
    logger.info(f"Saved FT last checkpoint: {ckpt_last}")
    return {"best_metric": best_metric, "best_score": best_score, "best_epoch": best_epoch}


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/jepa_user_tower")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Protocol knobs
    ap.add_argument("--vocab_from_all_splits", action="store_true",
                    help="If set, build vocab by adding tokens from valid/test too (not leakage-safe). Default: train-only vocab with UNK.")
    ap.add_argument("--drop_oov_label", action="store_true",
                    help="If set and train-only vocab, drop valid/test trails whose target POI is UNK.")

    # sequence
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--min_len_ssl", type=int, default=32)
    ap.add_argument("--ssl_use_full_prob", type=float, default=0.1)

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--pred_layers", type=int, default=3)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    # SSL
    ap.add_argument("--ssl_steps", type=int, default=5000)
    ap.add_argument("--ssl_batch", type=int, default=64)
    ap.add_argument("--ssl_lr", type=float, default=2e-4)
    ap.add_argument("--ssl_wd", type=float, default=1e-2)
    ap.add_argument("--mask_ratio", type=float, default=0.45)
    ap.add_argument("--mean_span_len", type=int, default=6)
    ap.add_argument("--ema_momentum", type=float, default=0.99)
    ap.add_argument("--ssl_log_every", type=int, default=100)
    ap.add_argument("--ssl_save_every", type=int, default=1000)

    # JEPA loss
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_var", type=float, default=1.0)
    ap.add_argument("--w_cov", type=float, default=0.04)
    ap.add_argument("--target_std", type=float, default=1.0)

    # finetune
    ap.add_argument("--ft_epochs", type=int, default=6)
    ap.add_argument("--ft_batch", type=int, default=256)
    ap.add_argument("--ft_lr", type=float, default=3e-4)
    ap.add_argument("--ft_wd", type=float, default=1e-2)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--ft_log_every", type=int, default=50)
    ap.add_argument("--best_metric", type=str, default="ndcg@10", choices=["acc@10", "ndcg@10", "ndcg@5", "acc@5"])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    run_dir = make_run_dir(args.out_dir, args.seed)
    logger = setup_logger(run_dir)
    set_seed(args.seed)

    logger.info("Starting run")
    logger.info(f"Run dir: {run_dir}")
    save_json(vars(args), os.path.join(run_dir, "config.json"))

    # Load CSVs
    train_df = load_checkins(args.train_csv)
    valid_df = load_checkins(args.valid_csv)
    test_df  = load_checkins(args.test_csv)
    logger.info(f"Loaded CSVs: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # Vocab: PAD, UNK, SEP for venues and cats; PAD, UNK for users.
    venue_vocab = Vocab(["<PAD>", "<UNK>", "<SEP>"])
    user_vocab  = Vocab(["<PAD>", "<UNK>"])
    cat_vocab   = Vocab(["<PAD>", "<UNK>", "<SEP_CAT>"])

    PAD_V = venue_vocab.stoi["<PAD>"]
    UNK_V = venue_vocab.stoi["<UNK>"]
    SEP_V = venue_vocab.stoi["<SEP>"]
    PAD_C = cat_vocab.stoi["<PAD>"]
    UNK_C = cat_vocab.stoi["<UNK>"]
    SEP_C = cat_vocab.stoi["<SEP_CAT>"]

    # Build trails
    add_train = True
    add_other = args.vocab_from_all_splits

    train_trails = build_trails(train_df, venue_vocab, user_vocab, cat_vocab, add_new_tokens=add_train)
    valid_trails = build_trails(valid_df, venue_vocab, user_vocab, cat_vocab, add_new_tokens=add_other)
    test_trails  = build_trails(test_df,  venue_vocab, user_vocab, cat_vocab, add_new_tokens=add_other)

    logger.info(f"Vocab sizes: venues={len(venue_vocab)} users={len(user_vocab)} cats={len(cat_vocab)}")
    logger.info(f"Trails: train={len(train_trails)} valid={len(valid_trails)} test={len(test_trails)}")

    # If leakage-safe (train-only vocab), we must rebuild valid/test using get() only (UNK) not add()
    if not args.vocab_from_all_splits:
        valid_trails = build_trails(valid_df, venue_vocab, user_vocab, cat_vocab, add_new_tokens=False)
        test_trails  = build_trails(test_df,  venue_vocab, user_vocab, cat_vocab, add_new_tokens=False)

    # Build per-user TRAIN histories with SEP between trails (TRAIN only -> no leakage)
    user_hist = build_user_histories_from_train_trails(train_trails, sep_venue_id=SEP_V, sep_cat_id=SEP_C)
    logger.info(f"Built user histories (train only): #users_with_history={len(user_hist)}")

    stats = {
        "vocab_sizes": {"venues": len(venue_vocab), "users": len(user_vocab), "cats": len(cat_vocab)},
        "trail_len_stats": {
            "train": trail_length_stats(train_trails),
            "valid": trail_length_stats(valid_trails),
            "test":  trail_length_stats(test_trails),
        },
        "user_hist_count": len(user_hist),
        "protocol": {
            "vocab_from_all_splits": args.vocab_from_all_splits,
            "drop_oov_label": args.drop_oov_label,
            "ssl_on": "train_user_histories_with_sep",
            "finetune_on": "train_trails_prefix_to_last",
            "eval_on": "valid/test_trails_prefix_to_last",
        }
    }

    # Supervised datasets with optional dropping of OOV target labels
    sup_train = NextPOIDataset(train_trails, args.max_len, UNK_V, drop_oov_label=False)
    sup_valid = NextPOIDataset(valid_trails, args.max_len, UNK_V, drop_oov_label=args.drop_oov_label)
    sup_test  = NextPOIDataset(test_trails,  args.max_len, UNK_V, drop_oov_label=args.drop_oov_label)

    stats["supervised_counts"] = {
        "train": len(sup_train),
        "valid": len(sup_valid),
        "test": len(sup_test),
    }
    save_json(stats, os.path.join(run_dir, "data_stats.json"))
    logger.info(f"Trail length stats: {stats['trail_len_stats']}")
    logger.info(f"Supervised counts (after drop_oov_label if enabled): {stats['supervised_counts']}")

    # DataLoaders
    ssl_ds = UserHistorySSLDataset(user_hist, max_len=args.max_len, min_len=args.min_len_ssl,
                                  use_full_prob=args.ssl_use_full_prob, seed=args.seed)
    ssl_loader = DataLoader(
        ssl_ds, batch_size=args.ssl_batch, shuffle=True, num_workers=0,
        collate_fn=lambda b: ssl_collate(b, args.max_len),
        drop_last=True
    )

    train_loader = DataLoader(
        sup_train, batch_size=args.ft_batch, shuffle=True, num_workers=0,
        collate_fn=lambda b: sup_collate(b, args.max_len), drop_last=True
    )
    valid_loader = DataLoader(
        sup_valid, batch_size=max(256, args.ft_batch), shuffle=False, num_workers=0,
        collate_fn=lambda b: sup_collate(b, args.max_len)
    )
    test_loader = DataLoader(
        sup_test, batch_size=max(256, args.ft_batch), shuffle=False, num_workers=0,
        collate_fn=lambda b: sup_collate(b, args.max_len)
    )

    # Model
    model = JEPAUserTower(
        num_venues=len(venue_vocab),
        num_users=len(user_vocab),
        num_cats=len(cat_vocab),
        d_model=args.d_model,
        enc_layers=args.enc_layers,
        pred_layers=args.pred_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_len=args.max_len,
    )

    loss_cfg = JEPALossConfig(args.w_mse, args.w_var, args.w_cov, args.target_std)

    # SSL pretrain
    logger.info("== SSL pretraining (JEPA) on TRAIN user histories with <SEP> ==")
    pretrain_ssl(
        model=model, loader=ssl_loader, device=args.device,
        steps=args.ssl_steps, lr=args.ssl_lr, weight_decay=args.ssl_wd,
        mask_ratio=args.mask_ratio, mean_span_len=args.mean_span_len,
        ema_momentum=args.ema_momentum, loss_cfg=loss_cfg,
        grad_clip=args.grad_clip, log_every=args.ssl_log_every,
        save_every=args.ssl_save_every, run_dir=run_dir, logger=logger,
        sep_venue_id=SEP_V, args_dict=vars(args)
    )

    # Finetune
    head = NextPOIHead(d_model=model.d_model, num_venues=len(venue_vocab))
    logger.info("== Fine-tuning on TRAIN trails (prefix -> last POI) ==")
    best = finetune_next_poi(
        model=model, head=head,
        train_loader=train_loader, valid_loader=valid_loader,
        device=args.device, epochs=args.ft_epochs,
        lr=args.ft_lr, weight_decay=args.ft_wd,
        freeze_encoder=args.freeze_encoder,
        pool=args.pool, log_every=args.ft_log_every,
        run_dir=run_dir, logger=logger,
        best_metric=args.best_metric, args_dict=vars(args)
    )
    logger.info(f"Best finetune: {best}")

    # Test evaluation
    test_metrics = evaluate_next_poi(model, head, test_loader, args.device, pool=args.pool)
    logger.info(f"== TEST METRICS == {test_metrics}")
    save_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))

    # Human readable summary
    with open(os.path.join(run_dir, "README_run.txt"), "w") as f:
        f.write("JEPA User Tower (SSL per-user histories with <SEP>, FT per-trail)\n\n")
        f.write("CONFIG:\n")
        f.write(json.dumps(vars(args), indent=2))
        f.write("\n\nSTATS:\n")
        f.write(json.dumps(stats, indent=2))
        f.write("\n\nBEST:\n")
        f.write(json.dumps(best, indent=2))
        f.write("\n\nTEST:\n")
        f.write(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()