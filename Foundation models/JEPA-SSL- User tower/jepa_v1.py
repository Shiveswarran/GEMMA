"""
JEPA (I-JEPA-style) User Tower for Next-POI Prediction

- Each trail_id defines a trajectory (sorted by timestamp).
- For train/valid/test: the LAST POI (venue_id) in the trail is the target.
  The prefix POIs are the known trajectory context.

Pipeline:
1) Build vocabularies (venue_id, user_id, category_id_code) from train (+ optional valid/test).
2) SSL pretrain (JEPA): masked latent prediction on sequences (span masking), EMA target encoder,
   variance/cov regularization to reduce collapse.
3) Supervised fine-tune: next-POI classification from prefix to last venue_id.
4) Evaluate: Acc@1/5/10 and NDCG@5/10 on valid/test.

Run example:
  python run_jepa_user_tower.py \
    --train_csv data/new_york/new_york_checkins_train.csv \
    --valid_csv data/new_york/new_york_checkins_validation.csv \
    --test_csv  data/new_york/new_york_checkins_test.csv \
    --out_dir   outputs/jepa_user_tower_ny \
    --device    cuda

Notes:
- This focuses ONLY on User Tower: uses POI IDs + (lat,lon) + time features + user_id embedding.
- Later replace venue embeddings with fusion of semantic/knowledge towers.
"""

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import logging
from datetime import datetime
from pathlib import Path
# ============================================================
# Repro
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics: Acc@k and NDCG@k
# ============================================================
@torch.no_grad()
def acc_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    logits: (B, V) V= venues (POIs)
    labels: (B,) B= batch size each in [0,V-1]
    """
    topk = torch.topk(logits, k=k, dim=-1).indices  # (B,k)
    hit = (topk == labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return hit


@torch.no_grad()
def ndcg_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    NDCG for single ground-truth item:
      DCG = 1 / log2(rank+1) if label is in top-k else 0
      IDCG = 1 (best possible with single item at rank 1)
      NDCG = DCG / IDCG = DCG
    """
    topk = torch.topk(logits, k=k, dim=-1).indices  # (B,k)
    # ranks: 0..k-1
    matches = (topk == labels.unsqueeze(-1))  # (B,k) bool
    # find rank of the match, if any
    # If multiple (shouldn't happen), take first.
    ranks = torch.argmax(matches.int(), dim=-1)  # (B,)
    has = matches.any(dim=-1)  # (B,)
    # DCG
    dcg = torch.zeros_like(labels, dtype=torch.float)
    # rank is 0-based, formula uses rank+2 in log2 for (rank+1) in DCG definition where rank starts at 1
    dcg[has] = 1.0 / torch.log2((ranks[has].float() + 2.0))
    return dcg.mean().item()

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

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File
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

#HELPER
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

def save_checkpoint(path: str, model: nn.Module, head: Optional[nn.Module], args_dict: Dict, extra: Dict):
    payload = {
        "model": model.state_dict(),
        "args": args_dict,
        "extra": extra,
    }
    if head is not None:
        payload["head"] = head.state_dict()
    torch.save(payload, path)


# ============================================================
# Vocab
# ============================================================
class Vocab:
    """
    Maps tokens (strings/ints) -> contiguous ids.
    Reserves 0 for PAD. --> POI IDs can be huge and sparse, so need to map them to dense contiguous indices.
    """
    def __init__(self, pad_token="<PAD>"):
        self.pad_token = pad_token
        self.stoi: Dict[str, int] = {pad_token: 0}
        self.itos: List[str] = [pad_token]

    def add(self, token) -> int:
        s = str(token)
        if s not in self.stoi:
            self.stoi[s] = len(self.itos)
            self.itos.append(s)
        return self.stoi[s]

    def get(self, token) -> int:
        return self.stoi.get(str(token), 0)

    def __len__(self):
        return len(self.itos)


# ============================================================
# Data loading: build trajectories per trail_id
# ============================================================
REQUIRED_COLS = ["trail_id", "user_id", "venue_id", "latitude", "longitude", "timestamp"]


def _safe_float(x, default=0.0) -> float: #blank fields, NaNs, numeric strings
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        # sometimes comes as "331" or "331.0"
        return int(float(x))
    except Exception:
        return default


def load_checkins(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Optional category code column
    if "venue_category_id_code" not in df.columns:
        df["venue_category_id_code"] = 0

    # Parse timestamp
    # Your sample shows "2012-07-13 01:30:00"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Drop rows with bad timestamps or missing essentials
    df = df.dropna(subset=["trail_id", "user_id", "venue_id", "timestamp"])
    return df


def build_trails(
    df: pd.DataFrame,
    venue_vocab: Vocab,
    user_vocab: Vocab,
    cat_vocab: Vocab,
) -> Dict[str, Dict]:
    """
    Returns:
      trails[trail_id] = {
        "user": user_idx,
        "venue": [v1..vL],
        "cat":   [c1..cL],
        "lat":   [..],
        "lon":   [..],
        "hour":  [..],
        "dow":   [..],
        "dtb":   [..]  # time-delta bucket from prev checkin
      }
    """
    trails = {}
    # Sort by trail_id then timestamp
    df = df.sort_values(["trail_id", "timestamp"], ascending=True)

    # groupby trail
    for tid, g in df.groupby("trail_id", sort=False):
        g = g.sort_values("timestamp")
        user_id = g["user_id"].iloc[0]
        u = user_vocab.add(user_id)

        venues = []
        cats = []
        lats = []
        lons = []
        hours = []
        dows = []
        dtb = []

        prev_ts = None
        for _, row in g.iterrows():
            v = venue_vocab.add(row["venue_id"])
            c = cat_vocab.add(_safe_int(row.get("venue_category_id_code", 0), 0))

            ts = row["timestamp"]
            lat = _safe_float(row.get("latitude", 0.0), 0.0)
            lon = _safe_float(row.get("longitude", 0.0), 0.0)

            hour = int(ts.hour) if pd.notna(ts) else 0
            dow = int(ts.dayofweek) if pd.notna(ts) else 0

            if prev_ts is None or pd.isna(prev_ts) or pd.isna(ts):
                delta_min = 0.0
            else:
                delta_min = max(0.0, (ts - prev_ts).total_seconds() / 60.0)

            # bucket time gap (minutes) into ~log bins
            # 0-5, 5-15, 15-30, 30-60, 1-2h, 2-4h, 4-8h, 8-24h, 1-3d, >3d
            dt_bucket = time_delta_bucket(delta_min)

            venues.append(v)
            cats.append(cat_vocab.get(c))
            lats.append(lat)
            lons.append(lon)
            hours.append(hour)
            dows.append(dow)
            dtb.append(dt_bucket)

            prev_ts = ts

        if len(venues) < 2:
            # cannot do next-POI if only one checkin
            continue

        trails[str(tid)] = {
            "user": u,
            "venue": venues,
            "cat": cats,
            "lat": lats,
            "lon": lons,
            "hour": hours,
            "dow": dows,
            "dtb": dtb,
        }

    return trails


def time_delta_bucket(delta_min: float) -> int:  #Raw continous time gaps have heavy tails
    # returns small integer bucket id
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


# ============================================================
# Dataset for SSL (JEPA)
# ============================================================
def make_span_mask(seq_len: int, mask_ratio: float, mean_span_len: int, rng: random.Random) -> torch.BoolTensor: 
    #Generate contigous masked blocks in a sequence ==> Missing a sub trajectory (commute chain/ lunch loop)
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


class JEPASSLDataset(Dataset):
    """
    Samples windows from trajectories for SSL.
    Each item returns a windowed sequence with features. (not necessariliy whole trail each time)
    """
    def __init__(
        self,
        trails: Dict[str, Dict],
        max_len: int = 64,
        min_len: int = 16,
        use_full_trail_prob: float = 0.2,
        seed: int = 0,
    ):
        self.items = list(trails.values())
        self.max_len = max_len
        self.min_len = min_len
        self.use_full_trail_prob = use_full_trail_prob
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        tr = self.items[idx]
        L = len(tr["venue"])

        # choose a window --> window sampling
        if self.rng.random() < self.use_full_trail_prob:
            start = 0
            end = L
        else:
            wlen = self.rng.randint(min(self.min_len, L), min(self.max_len, L))
            start = self.rng.randint(0, L - wlen)
            end = start + wlen

        return slice_trail(tr, start, end) #dict of tensors : venue/cat/latlon/hr/dow/dtb and user


def slice_trail(tr: Dict, start: int, end: int) -> Dict[str, torch.Tensor]:
    # returns dict of tensors for a subsequence
    return {
        "user": torch.tensor(tr["user"], dtype=torch.long),
        "venue": torch.tensor(tr["venue"][start:end], dtype=torch.long),
        "cat": torch.tensor(tr["cat"][start:end], dtype=torch.long),
        "latlon": torch.tensor(np.stack([tr["lat"][start:end], tr["lon"][start:end]], axis=-1), dtype=torch.float),
        "hour": torch.tensor(tr["hour"][start:end], dtype=torch.long),
        "dow": torch.tensor(tr["dow"][start:end], dtype=torch.long),
        "dtb": torch.tensor(tr["dtb"][start:end], dtype=torch.long),
    }


def pad_1d(x: torch.Tensor, max_len: int, pad_val: int = 0) -> torch.Tensor:
    # x: (T,) Transformers need fixed length batches or padding mask
    T = x.size(0)
    if T >= max_len:
        return x[:max_len]
    pad = torch.full((max_len - T,), pad_val, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def pad_2d(x: torch.Tensor, max_len: int, pad_val: float = 0.0) -> torch.Tensor:
    # x: (T,2)
    T = x.size(0)
    if T >= max_len:
        return x[:max_len]
    pad = torch.full((max_len - T, x.size(1)), pad_val, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def ssl_collate(batch: List[Dict], max_len: int):
    # batch entries have variable length
    users = torch.stack([b["user"] for b in batch], dim=0)  # (B,)
    lens = torch.tensor([b["venue"].size(0) for b in batch], dtype=torch.long)  # (B,)

    venue = torch.stack([pad_1d(b["venue"], max_len, 0) for b in batch], dim=0)  # (B,T)
    cat = torch.stack([pad_1d(b["cat"], max_len, 0) for b in batch], dim=0)
    hour = torch.stack([pad_1d(b["hour"], max_len, 0) for b in batch], dim=0)
    dow = torch.stack([pad_1d(b["dow"], max_len, 0) for b in batch], dim=0)
    dtb = torch.stack([pad_1d(b["dtb"], max_len, 0) for b in batch], dim=0)
    latlon = torch.stack([pad_2d(b["latlon"], max_len, 0.0) for b in batch], dim=0)  # (B,T,2)

    # padding mask: True where padded
    pad_mask = torch.arange(max_len).unsqueeze(0) >= lens.unsqueeze(1)  # (B,T)
    return {
        "user": users,
        "venue": venue,
        "cat": cat,
        "hour": hour,
        "dow": dow,
        "dtb": dtb,
        "latlon": latlon,
        "pad_mask": pad_mask,
        "lens": lens,
    }


# ============================================================
# Dataset for supervised next-POI (prefix -> last item)
# ============================================================
class NextPOIDataset(Dataset):
    """
    For each trail:
      prefix = all but last (or truncated to max_len) --> recent history generally predicts next POI best (Markovian bias)
      label  = last venue_id
    """
    def __init__(self, trails: Dict[str, Dict], max_len: int = 64):
        self.items = list(trails.values())
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        tr = self.items[idx]
        L = len(tr["venue"])
        label = tr["venue"][L - 1]

        # prefix
        start = max(0, (L - 1) - self.max_len)
        end = L - 1
        x = slice_trail(tr, start, end)
        x["label"] = torch.tensor(label, dtype=torch.long)
        return x


def sup_collate(batch: List[Dict], max_len: int):
    users = torch.stack([b["user"] for b in batch], dim=0)  # (B,)
    lens = torch.tensor([b["venue"].size(0) for b in batch], dtype=torch.long)  # (B,)

    venue = torch.stack([pad_1d(b["venue"], max_len, 0) for b in batch], dim=0)
    cat = torch.stack([pad_1d(b["cat"], max_len, 0) for b in batch], dim=0)
    hour = torch.stack([pad_1d(b["hour"], max_len, 0) for b in batch], dim=0)
    dow = torch.stack([pad_1d(b["dow"], max_len, 0) for b in batch], dim=0)
    dtb = torch.stack([pad_1d(b["dtb"], max_len, 0) for b in batch], dim=0)
    latlon = torch.stack([pad_2d(b["latlon"], max_len, 0.0) for b in batch], dim=0)

    pad_mask = torch.arange(max_len).unsqueeze(0) >= lens.unsqueeze(1)  # (B,T)
    labels = torch.stack([b["label"] for b in batch], dim=0)  # (B,)
    return {
        "user": users,
        "venue": venue,
        "cat": cat,
        "hour": hour,
        "dow": dow,
        "dtb": dtb,
        "latlon": latlon,
        "pad_mask": pad_mask,
        "lens": lens,
        "label": labels,
    }


# ============================================================
# Model: User Tower JEPA (context encoder + target encoder + predictor)
# Includes spatio-temporal + user embedding
# ============================================================
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class UserSequenceEncoder(nn.Module):
    """
    Encodes a sequence of check-ins into token latents (B,T,D).
    Input features:
      - venue_id (categorical)
      - category_code (categorical)
      - hour, day-of-week (categorical)
      - time-delta bucket (categorical)
      - lat/lon (continuous)
      - user_id (categorical) -> broadcast across tokens (user memory across trails)
    """
    def __init__(
        self,
        num_venues: int,
        num_users: int,
        num_cats: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
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

        # lat/lon -> D via MLP
        self.geo_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # A light feature fusion layer (helps stabilize)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        user: torch.Tensor,      # (B,)
        venue: torch.Tensor,     # (B,T)
        cat: torch.Tensor,       # (B,T)
        hour: torch.Tensor,      # (B,T)
        dow: torch.Tensor,       # (B,T)
        dtb: torch.Tensor,       # (B,T)
        latlon: torch.Tensor,    # (B,T,2)
        pad_mask: Optional[torch.Tensor] = None,  # (B,T) True where padded
    ) -> torch.Tensor:
        B, T = venue.shape

        v = self.venue_emb(venue)          # (B,T,D)
        c = self.cat_emb(cat)
        h = self.hour_emb(hour)
        d = self.dow_emb(dow)
        g = self.dtb_emb(dtb)
        geo = self.geo_mlp(latlon)         # (B,T,D)

        u = self.user_emb(user).unsqueeze(1).expand(B, T, self.d_model)  # (B,T,D)

        x = v + c + h + d + g + geo + u
        x = self.fuse(x)
        x = self.pos(x)
        x = self.tr(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)
        return x  # (B,T,D)


class PredictorTransformer(nn.Module):
    """
    Predictor that takes context latents with masked positions replaced by a learned mask token.
    Outputs predicted latents for all tokens; loss computed only on masked tokens.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1, max_len: int = 256):
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
        """
        ctx_latents: (B,T,D)
        mask: (B,T) True = masked token to predict
        """
        B, T, D = ctx_latents.shape
        m = mask.unsqueeze(-1)  # (B,T,1)
        mask_tok = self.mask_token.expand(B, T, D)
        x = torch.where(m, mask_tok, ctx_latents)
        x = self.pos(x)
        x = self.tr(x)
        x = self.ln(x)
        x = self.proj(x)
        return x  # (B,T,D)


def vicreg_variance_loss(z: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0) -> torch.Tensor:
    #prevent collapse, without regularization JEPA can converge to constant vectors
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
    def __init__(
        self,
        num_venues: int,
        num_users: int,
        num_cats: int,
        d_model: int = 256,
        enc_layers: int = 4,
        pred_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
        dtb_bins: int = 10,
    ):
        super().__init__()
        self.context_encoder = UserSequenceEncoder(
            num_venues=num_venues,
            num_users=num_users,
            num_cats=num_cats,
            d_model=d_model,
            nhead=nhead,
            num_layers=enc_layers,
            dropout=dropout,
            max_len=max_len,
            dtb_bins=dtb_bins,
        )
        self.target_encoder = UserSequenceEncoder(
            num_venues=num_venues,
            num_users=num_users,
            num_cats=num_cats,
            d_model=d_model,
            nhead=nhead,
            num_layers=enc_layers,
            dropout=dropout,
            max_len=max_len,
            dtb_bins=dtb_bins,
        )
        self.predictor = PredictorTransformer(
            d_model=d_model, nhead=nhead, num_layers=pred_layers, dropout=dropout, max_len=max_len
        )

        # init target = context
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.d_model = d_model

    @torch.no_grad()
    def ema_update_target(self, momentum: float):
        for pt, ps in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)

    def forward_ssl(
        self,
        batch: Dict[str, torch.Tensor],
        mask: torch.BoolTensor,
        loss_cfg: JEPALossConfig,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        batch contains:
          user (B,), venue/cat/hour/dow/dtb (B,T), latlon (B,T,2), pad_mask (B,T)
        mask: (B,T) True=masked positions (only among non-padding positions)
        """
        user = batch["user"]
        venue = batch["venue"]
        cat = batch["cat"]
        hour = batch["hour"]
        dow = batch["dow"]
        dtb = batch["dtb"]
        latlon = batch["latlon"]
        pad_mask = batch["pad_mask"]

        ctx = self.context_encoder(user, venue, cat, hour, dow, dtb, latlon, pad_mask)  # (B,T,D)
        with torch.no_grad():
            tgt = self.target_encoder(user, venue, cat, hour, dow, dtb, latlon, pad_mask)  # (B,T,D)

        pred = self.predictor(ctx, mask=mask)  # (B,T,D)

        B, T, D = pred.shape
        flat_mask = mask.view(-1)
        pred_flat = pred.view(-1, D)[flat_mask]  # (Nmask,D)
        tgt_flat = tgt.view(-1, D)[flat_mask]    # (Nmask,D)

        if pred_flat.numel() == 0:
            zero = torch.tensor(0.0, device=pred.device)
            return zero, {"mse": zero, "var": zero, "cov": zero}

        mse = F.mse_loss(pred_flat, tgt_flat)
        var = vicreg_variance_loss(pred_flat, target_std=loss_cfg.target_std)
        cov = vicreg_covariance_loss(pred_flat)

        loss = loss_cfg.mask_mse_weight * mse + loss_cfg.var_weight * var + loss_cfg.cov_weight * cov
        return loss, {"mse": mse.detach(), "var": var.detach(), "cov": cov.detach()}

    def pooled_user_embedding(self, batch: Dict[str, torch.Tensor], pool: str = "mean") -> torch.Tensor:
        """
        Returns (B,D) user representation for the prefix sequence.
        Mask/padding is respected in mean pooling.
        """
        user = batch["user"]
        venue = batch["venue"]
        cat = batch["cat"]
        hour = batch["hour"]
        dow = batch["dow"]
        dtb = batch["dtb"]
        latlon = batch["latlon"]
        pad_mask = batch["pad_mask"]

        z = self.context_encoder(user, venue, cat, hour, dow, dtb, latlon, pad_mask)  # (B,T,D)

        if pool == "mean":
            # mean over non-pad tokens
            keep = (~pad_mask).unsqueeze(-1).float()  # (B,T,1)
            denom = keep.sum(dim=1).clamp(min=1.0)    # (B,1)
            return (z * keep).sum(dim=1) / denom      # (B,D)
        elif pool == "last":
            # take last non-pad token embedding
            lens = batch["lens"]  # (B,)
            idx = (lens - 1).clamp(min=0)
            return z[torch.arange(z.size(0), device=z.device), idx, :]  # (B,D)
        else:
            raise ValueError("pool must be 'mean' or 'last'")


class NextPOIHead(nn.Module):
    def __init__(self, d_model: int, num_venues: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.out = nn.Linear(d_model, num_venues)

    def forward(self, user_embed: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(user_embed))


# ============================================================
# Training loops
# ============================================================
def build_ssl_mask(
    pad_mask: torch.BoolTensor,
    mask_ratio: float,
    mean_span_len: int,
    rng: random.Random,
) -> torch.BoolTensor:
    """
    Create per-sample span mask, but ensure we do NOT mask padded tokens.
    pad_mask: (B,T) True where padding
    """
    B, T = pad_mask.shape
    masks = []
    for b in range(B):
        # effective length
        L = int((~pad_mask[b]).sum().item())
        if L <= 1:
            masks.append(torch.zeros(T, dtype=torch.bool))
            continue
        m = make_span_mask(L, mask_ratio, mean_span_len, rng)  # (L,)
        # extend to T with False on padding
        full = torch.zeros(T, dtype=torch.bool)
        full[:L] = m
        masks.append(full)
    return torch.stack(masks, dim=0)  # (B,T)


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
    logger, run_dir: str, save_every: int = 500
):
    model.to(device)
    model.train()

    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    rng = random.Random(0)
    it = iter(loader)

    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        ssl_csv = os.path.join(run_dir, "train_ssl_log.csv")
        ssl_header = ["step", "loss", "mse", "var", "cov", "lr", "time_sec"]

        t_step0 = time.time()

        mask = build_ssl_mask(batch["pad_mask"], mask_ratio, mean_span_len, rng).to(device)

        loss, m = model.forward_ssl(batch, mask=mask, loss_cfg=loss_cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        with torch.no_grad():
            model.ema_update_target(momentum=ema_momentum)

        if step % log_every == 0:
            print(
                f"[SSL] step={step:6d}/{steps} loss={loss.item():.4f} "
                f"mse={m['mse'].item():.4f} var={m['var'].item():.4f} cov={m['cov'].item():.4f}"
            )

        lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t_step0

        if step % log_every == 0:
            logger.info(
                f"[SSL] step={step}/{steps} loss={loss.item():.4f} mse={m['mse'].item():.4f} "
                f"var={m['var'].item():.4f} cov={m['cov'].item():.4f} lr={lr:.2e} dt={elapsed:.2f}s"
            )

        append_csv_row(
            ssl_csv, ssl_header,
            {
                "step": step,
                "loss": float(loss.item()),
                "mse": float(m["mse"].item()),
                "var": float(m["var"].item()),
                "cov": float(m["cov"].item()),
                "lr": lr,
                "time_sec": elapsed,
            }
        )

        if save_every > 0 and step % save_every == 0:
            ckpt_path = os.path.join(run_dir, "checkpoints", f"ckpt_ssl_step{step}.pt")
            save_checkpoint(
                ckpt_path, model=model, head=None,
                args_dict={},  # you can pass vars(args) from main; or store separately
                extra={"step": step}
            )
            logger.info(f"Saved SSL checkpoint: {ckpt_path}")

        t_step0 = time.time()


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
    logger, run_dir: str, best_metric: str = "ndcg@10"
):
    model.to(device)
    head.to(device)

    if freeze_encoder:
        for p in model.context_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.context_encoder.parameters():
            p.requires_grad = True

    params = list(head.parameters()) + [p for p in model.context_encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best = {"acc@10": -1.0, "epoch": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()

        running = 0.0
        for it, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch["label"]
            
            train_csv = os.path.join(run_dir, "train_ft_log.csv")
            train_header = ["epoch", "iter", "loss", "acc@1", "acc@5", "acc@10", "ndcg@5", "ndcg@10", "lr", "time_sec"]

            u = model.pooled_user_embedding(batch, pool=pool)  # (B,D)
            logits = head(u)                                   # (B,V)

            loss = F.cross_entropy(logits, label)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            if it % log_every == 0:
                avg = running / log_every
                running = 0.0
                a1 = acc_at_k(logits.detach(), label.detach(), 1)
                a5 = acc_at_k(logits.detach(), label.detach(), 5)
                a10 = acc_at_k(logits.detach(), label.detach(), 10)
                n5 = ndcg_at_k(logits.detach(), label.detach(), 5)
                n10 = ndcg_at_k(logits.detach(), label.detach(), 10)
                print(f"[FT][ep={epoch}] it={it:5d} loss={avg:.4f} acc1={a1:.3f} acc5={a5:.3f} acc10={a10:.3f} ndcg5={n5:.3f} ndcg10={n10:.3f}")

        # valid
        metrics = evaluate_next_poi(model, head, valid_loader, device, pool=pool)
        print(f"[VAL][ep={epoch}] {metrics}")

        valid_csv = os.path.join(run_dir, "valid_log.csv")
        valid_header = ["epoch", "acc@1", "acc@5", "acc@10", "ndcg@5", "ndcg@10"]

        append_csv_row(valid_csv, valid_header, {"epoch": epoch, **metrics})
        logger.info(f"[VAL] epoch={epoch} metrics={metrics}")

        if metrics["acc@10"] > best["acc@10"]:
            best = {"acc@10": metrics["acc@10"], "epoch": epoch}

        metric_val = metrics.get(best_metric, metrics["acc@10"])
        if metric_val > best_score:
            best_score = metric_val
            best_epoch = epoch
            ckpt_best = os.path.join(run_dir, "checkpoints", "ckpt_ft_best.pt")
            save_checkpoint(ckpt_best, model=model, head=head, args_dict={}, extra={"epoch": epoch, "best_metric": best_metric,     "best_score": best_score})
            logger.info(f"Saved BEST finetune checkpoint: {ckpt_best} ({best_metric}={best_score:.4f})")

    return best


@torch.no_grad()
def evaluate_next_poi(model: JEPAUserTower, head: NextPOIHead, loader: DataLoader, device: str, pool: str) -> Dict[str, float]:
    model.eval()
    head.eval()
    all_acc1, all_acc5, all_acc10 = [], [], []
    all_n5, all_n10 = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        label = batch["label"]
        u = model.pooled_user_embedding(batch, pool=pool)
        logits = head(u)

        all_acc1.append(acc_at_k(logits, label, 1))
        all_acc5.append(acc_at_k(logits, label, 5))
        all_acc10.append(acc_at_k(logits, label, 10))
        all_n5.append(ndcg_at_k(logits, label, 5))
        all_n10.append(ndcg_at_k(logits, label, 10))

    return {
        "acc@1": float(np.mean(all_acc1)) if all_acc1 else 0.0,
        "acc@5": float(np.mean(all_acc5)) if all_acc5 else 0.0,
        "acc@10": float(np.mean(all_acc10)) if all_acc10 else 0.0,
        "ndcg@5": float(np.mean(all_n5)) if all_n5 else 0.0,
        "ndcg@10": float(np.mean(all_n10)) if all_n10 else 0.0,
    }


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

    # sequence limits
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--min_len_ssl", type=int, default=16)

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--pred_layers", type=int, default=3)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    # SSL
    ap.add_argument("--ssl_steps", type=int, default=3000)
    ap.add_argument("--ssl_batch", type=int, default=64)
    ap.add_argument("--ssl_lr", type=float, default=2e-4)
    ap.add_argument("--ssl_wd", type=float, default=1e-2)
    ap.add_argument("--mask_ratio", type=float, default=0.45)
    ap.add_argument("--mean_span_len", type=int, default=4)
    ap.add_argument("--ema_momentum", type=float, default=0.99)
    ap.add_argument("--ssl_log_every", type=int, default=100)

    # loss weights
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_var", type=float, default=1.0)
    ap.add_argument("--w_cov", type=float, default=0.04)
    ap.add_argument("--target_std", type=float, default=1.0)

    # Finetune
    ap.add_argument("--ft_epochs", type=int, default=6)
    ap.add_argument("--ft_batch", type=int, default=128)
    ap.add_argument("--ft_lr", type=float, default=3e-4)
    ap.add_argument("--ft_wd", type=float, default=1e-2)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--ft_log_every", type=int, default=50)

    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    run_dir = make_run_dir(args.out_dir, args.seed)
    logger = setup_logger(run_dir)
    logger.info("Starting run")
    logger.info(f"Run dir: {run_dir}")
    save_json(vars(args), os.path.join(run_dir, "config.json"))

    # 1) Load CSVs
    train_df = load_checkins(args.train_csv)
    valid_df = load_checkins(args.valid_csv)
    test_df = load_checkins(args.test_csv)

    logger.info(f"Loaded CSVs: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # 2) Build vocabs from TRAIN only (recommended). If you want closed-world vocab, include valid/test too.
    venue_vocab = Vocab()
    user_vocab = Vocab()
    cat_vocab = Vocab()

    train_trails = build_trails(train_df, venue_vocab, user_vocab, cat_vocab)
    valid_trails = build_trails(valid_df, venue_vocab, user_vocab, cat_vocab)
    test_trails = build_trails(test_df, venue_vocab, user_vocab, cat_vocab)

    print(f"#venues={len(venue_vocab)} #users={len(user_vocab)} #cats={len(cat_vocab)}")
    print(f"trails: train={len(train_trails)} valid={len(valid_trails)} test={len(test_trails)}")

    # Save vocab for later tower fusion
    with open(os.path.join(args.out_dir, "vocab_sizes.json"), "w") as f:
        json.dump(
            {"venues": len(venue_vocab), "users": len(user_vocab), "cats": len(cat_vocab)},
            f, indent=2
        )
    
    stats = {
        "vocab_sizes": {"venues": len(venue_vocab), "users": len(user_vocab), "cats": len(cat_vocab)},
        "trail_stats": {
            "train": trail_length_stats(train_trails),
            "valid": trail_length_stats(valid_trails),
            "test":  trail_length_stats(test_trails),
        },
    }
    save_json(stats, os.path.join(run_dir, "data_stats.json"))
    logger.info(f"Vocab sizes: {stats['vocab_sizes']}")
    logger.info(f"Trail length stats: {stats['trail_stats']}")

    # 3) Dataloaders
    ssl_ds = JEPASSLDataset(train_trails, max_len=args.max_len, min_len=args.min_len_ssl, seed=args.seed)
    ssl_loader = DataLoader(
        ssl_ds,
        batch_size=args.ssl_batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: ssl_collate(b, args.max_len),
        drop_last=True,
    )

    sup_train = NextPOIDataset(train_trails, max_len=args.max_len)
    sup_valid = NextPOIDataset(valid_trails, max_len=args.max_len)
    sup_test = NextPOIDataset(test_trails, max_len=args.max_len)

    train_loader = DataLoader(
        sup_train,
        batch_size=args.ft_batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: sup_collate(b, args.max_len),
        drop_last=True,
    )
    valid_loader = DataLoader(
        sup_valid,
        batch_size=max(256, args.ft_batch),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: sup_collate(b, args.max_len),
    )
    test_loader = DataLoader(
        sup_test,
        batch_size=max(256, args.ft_batch),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: sup_collate(b, args.max_len),
    )

    # 4) Model
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

    loss_cfg = JEPALossConfig(
        mask_mse_weight=args.w_mse,
        var_weight=args.w_var,
        cov_weight=args.w_cov,
        target_std=args.target_std,
    )

    # 5) SSL pretrain
    print("== SSL pretraining (JEPA) ==")
    t0 = time.time()
    pretrain_ssl(
        model=model,
        loader=ssl_loader,
        device=args.device,
        steps=args.ssl_steps,
        lr=args.ssl_lr,
        weight_decay=args.ssl_wd,
        mask_ratio=args.mask_ratio,
        mean_span_len=args.mean_span_len,
        ema_momentum=args.ema_momentum,
        loss_cfg=loss_cfg,
        grad_clip=args.grad_clip,
        log_every=args.ssl_log_every,
        logger=logger,
        run_dir = run_dir
    )
    print(f"SSL done in {(time.time()-t0)/60.0:.1f} min")

    # Save SSL model
    ssl_path = os.path.join(args.out_dir, "jepa_user_tower_ssl.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ssl_path)
    print("saved:", ssl_path)

    # 6) Fine-tune for next-POI classification
    head = NextPOIHead(d_model=model.d_model, num_venues=len(venue_vocab))

    print("== Fine-tuning for next-POI ==")
    best = finetune_next_poi(
        model=model,
        head=head,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=args.device,
        epochs=args.ft_epochs,
        lr=args.ft_lr,
        weight_decay=args.ft_wd,
        freeze_encoder=args.freeze_encoder,
        pool=args.pool,
        log_every=args.ft_log_every,
        logger=logger,
        run_dir = run_dir
    )
    print("best:", best)

    # Save finetuned
    ft_path = os.path.join(args.out_dir, "jepa_user_tower_finetuned.pt")
    torch.save({"model": model.state_dict(), "head": head.state_dict(), "args": vars(args)}, ft_path)
    print("saved:", ft_path)

    # 7) Evaluate on test
    test_metrics = evaluate_next_poi(model, head, test_loader, args.device, pool=args.pool)
    print("== TEST METRICS ==")
    print(test_metrics)

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    save_json(test_metrics, os.path.join(run_dir, "test_metrics.json"))
    logger.info(f"TEST METRICS: {test_metrics}")


if __name__ == "__main__":
    main()