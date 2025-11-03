#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Tower Next-POI (v6)
- Name Tower: frozen ModernBERT 512-d (precomputed).
- KG Tower: frozen KGE (RotatE/MuRE/ComplEx/TransE...), projected to 512-d.
- Fusion: small learnable projection over concat([name, kg]).
- User Tower: Graph Transformer over per-trajectory graphs with spatio-temporal,
  semantic, and category edges; edge dropout; attention bias.
- Training: sampled-softmax with uniform negatives (plug your hard negatives if desired).
- Eval: Acc@1/5/10, NDCG@5 on val/test. Best checkpoint saved.

Folder layout expected:
  runs/<city>/name_emb/{name_emb.npy, poi_index.json}
  runs/<city>/kg/{entity_embeddings.pt|.npy, entity2id.json (optional)}
  data/<city>/*_checkins_{train,validation,test}.csv
"""

import os, sys, json, math, logging
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Utils & Logging
# -------------------------

def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def accuracy_at_k(ranks, k):
    r = np.asarray(ranks, dtype=np.int64)
    return float((r <= k).mean()) if len(r) else 0.0

def ndcg_at_k(ranks, k):
    r = np.asarray(ranks, dtype=np.int64)
    return float(((r <= k).astype(np.float32) / np.log2(r + 1)).mean()) if len(r) else 0.0

# -------------------------
# Data: trajectories with δτ = 8 hours (strict)
# -------------------------

def build_trajectories_delta(df: pd.DataFrame, delta_hours=8) -> List[Tuple[int, list]]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    delta = timedelta(hours=delta_hours)
    trajs = []
    for uid, g in df.groupby("user_id"):
        current, last_t = [], None
        for r in g.itertuples(index=False):
            t = r.timestamp
            if (last_t is not None) and (t - last_t > delta):
                if len(current) > 1:
                    trajs.append((uid, current))
                current = []
            current.append(r)
            last_t = t
        if len(current) > 1:
            trajs.append((uid, current))
    return trajs

# -------------------------
# Name embeddings (frozen, precomputed)
# -------------------------

def load_name_bank(run_city_dir: str):
    name_dir = os.path.join(run_city_dir, "name_emb")
    emb = torch.from_numpy(np.load(os.path.join(name_dir, "name_emb.npy"))).float()
    poi_index = json.load(open(os.path.join(name_dir, "poi_index.json")))
    # old poi_meta may not have lat/lon; we rebuild from CSVs below if needed
    meta_path = os.path.join(name_dir, "poi_meta.parquet")
    meta = pd.read_parquet(meta_path) if os.path.exists(meta_path) else None
    return emb, poi_index, meta

# -------------------------
# Rebuild POI meta (lat/lon/category) from CSVs if needed
# -------------------------

def rebuild_poi_meta_if_needed(meta: Optional[pd.DataFrame],
                               train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                               run_city_dir: str):
    if meta is not None and {"venue_id","latitude","longitude","venue_category"}.issubset(set(meta.columns)):
        return meta  # already rich enough

    all_df = pd.concat([train, valid, test], ignore_index=True)
    all_df["venue_id"] = all_df["venue_id"].astype(str)
    # keep the latest non-null lat/lon/category
    all_df = all_df.sort_values(["venue_id","timestamp"])
    grp = all_df.groupby("venue_id").agg({
        "name":"last",
        "venue_category":"last",
        "latitude":"last",
        "longitude":"last",
        "venue_city":"last",
        "address":"last"
    }).reset_index()
    meta_path = os.path.join(run_city_dir, "name_emb", "poi_meta_rich.parquet")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    grp.to_parquet(meta_path)
    return grp

# -------------------------
# KG embeddings (align -> normalize -> project 256/whatever -> 512)
# Align rows to POI index order (fallback mean for missing)
# -------------------------

def load_kg_embeddings_aligned(
    kg_dir: str,
    poi_index: Dict[str, int],
    target_dim: int = 512,
    device: str = "cpu",
):
    pt = os.path.join(kg_dir, "entity_embeddings.pt")
    npy = os.path.join(kg_dir, "entity_embeddings.npy")
    if os.path.exists(pt):
        kg_raw = torch.load(pt, map_location="cpu")
        if isinstance(kg_raw, dict) and "emb" in kg_raw:
            kg_raw = kg_raw["emb"]
        kg_raw = torch.as_tensor(kg_raw, dtype=torch.float32)
    elif os.path.exists(npy):
        kg_raw = torch.from_numpy(np.load(npy)).float()
    else:
        raise FileNotFoundError("No KG embeddings found (.pt or .npy).")

    logging.info(f"[KG] Detected embedding dim={kg_raw.size(1)}")

    # Map POI::<venue_id> -> row index in kg_raw
    entity2id_path = os.path.join(kg_dir, "entity2id.json")
    e2i = json.load(open(entity2id_path)) if os.path.exists(entity2id_path) else {}
    # Build aligned bank [N_poi, d_raw] with mean fill for missing
    N = len(poi_index)
    d_raw = kg_raw.size(1)
    kg_aligned = torch.zeros(N, d_raw)
    # mean vec for unknown
    mean_vec = kg_raw.mean(0, keepdim=True)
    mapped = 0
    for vid, row in poi_index.items():
        j = e2i.get(f"POI::{vid}")
        if j is not None and j < kg_raw.size(0):
            kg_aligned[row] = kg_raw[j]
            mapped += 1
        else:
            kg_aligned[row] = mean_vec
    logging.info(f"[KG Align] mapped {mapped}/{N} POIs; filled {N-mapped} with mean vector.")

    # linear projector to 512-d (trainable)
    projector = nn.Linear(d_raw, target_dim, bias=True)
    # initialize to near-orthogonal mapping
    with torch.no_grad():
        W = torch.randn(target_dim, d_raw)
        W, _ = torch.linalg.qr(W.T)
        projector.weight.copy_(W)
        projector.bias.zero_()


    return kg_aligned.to(device), projector.to(device)

# -------------------------
# Dataset (returns ids + timestamps for graph edges)
# -------------------------

@dataclass
class Batch:
    uid: List[int]
    ctx_ids: torch.Tensor  # [B,T]
    ctx_ts: torch.Tensor   # [B,T] (unix seconds) padded 0
    tgt_id: torch.Tensor   # [B]

class PairDataset(Dataset):
    def __init__(self, trajectories, poi_index, ctx_len=24):
        self.vid2row = poi_index
        self.ctx_len = ctx_len
        self.items = []
        for uid, seq in trajectories:
            vids = [str(getattr(r, "venue_id")) for r in seq]
            if not all(v in self.vid2row for v in vids): 
                continue
            ctx = vids[:-1]; tgt = vids[-1]
            ts_ctx = [int(pd.Timestamp(getattr(r, "timestamp")).timestamp()) for r in seq[:-1]]
            self.items.append((uid, ctx, ts_ctx, tgt))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        uid, ctx, ts_ctx, tgt = self.items[idx]
        idxs = [self.vid2row[v] for v in ctx][-self.ctx_len:]
        tss  = ts_ctx[-self.ctx_len:]
        pad_n = self.ctx_len - len(idxs)
        if pad_n > 0:
            idxs = [-1]*pad_n + idxs
            tss  = [0]*pad_n  + tss
        return {
            "uid": uid,
            "ctx_ids": torch.tensor(idxs, dtype=torch.long),
            "ctx_ts":  torch.tensor(tss,  dtype=torch.long),
            "tgt_id":  torch.tensor(self.vid2row[tgt], dtype=torch.long),
        }

def collate(batch) -> Batch:
    return Batch(
        uid=[b["uid"] for b in batch],
        ctx_ids=torch.stack([b["ctx_ids"] for b in batch], dim=0),
        ctx_ts=torch.stack([b["ctx_ts"]  for b in batch], dim=0),
        tgt_id=torch.stack([b["tgt_id"]  for b in batch], dim=0),
    )

# -------------------------
# Graph Utilities
# -------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    # expects tensors in degrees
    R = 6371.0088
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dlat/2)**2 + torch.cos(torch.deg2rad(lat1))*torch.cos(torch.deg2rad(lat2))*torch.sin(dlon/2)**2
    return 2*R*torch.arcsin(torch.sqrt(a))

def build_adjacency(ctx_ids: torch.Tensor, ctx_ts: torch.Tensor,
                    lat: torch.Tensor, lon: torch.Tensor, cat: torch.Tensor,
                    name_bank: torch.Tensor,
                    spatial_km=1.0, semantic_tau=0.75, include_category=True,
                    device="cpu") -> torch.Tensor:
    """
    Returns attention bias mask of shape [B, T, T]:
      0.0 for allowed edges, -inf for disallowed.
    Allowed edges: self, temporal neighbors (|i-j|=1), spatially close,
                   semantically similar, same category (optional).
    """
    B, T = ctx_ids.shape
    pad = (ctx_ids < 0)
    # basic allowed: self, consecutive
    allow = torch.eye(T, device=device).bool().unsqueeze(0).repeat(B,1,1)
    consec = torch.zeros(B, T, T, dtype=torch.bool, device=device)
    idx = torch.arange(T, device=device)
    consec[:, idx[:-1], idx[1:]] = True
    consec[:, idx[1:], idx[:-1]] = True
    allow |= consec

    # spatial
    # gather lat/lon [B,T]
    lat_bt = torch.where(pad, torch.nan, lat.gather(0, ctx_ids.clamp_min(0)))
    lon_bt = torch.where(pad, torch.nan, lon.gather(0, ctx_ids.clamp_min(0)))
    # pairwise valid mask
    for b in range(B):
        valid = ~pad[b]
        if valid.sum() == 0: 
            continue
        li = lat_bt[b, valid][:, None].repeat(1, valid.sum())
        lj = lat_bt[b, valid][None, :].repeat(valid.sum(), 1)
        oi = lon_bt[b, valid][:, None].repeat(1, valid.sum())
        oj = lon_bt[b, valid][None, :].repeat(valid.sum(), 1)
        dist = haversine_km(li, oi, lj, oj)
        close = dist <= spatial_km
        # scatter back into [T,T]
        idxv = torch.where(valid)[0]
        allow[b, idxv[:,None], idxv[None,:]] |= close

    # semantic (cosine in name space)
    name_norm = F.normalize(name_bank, dim=-1)
    for b in range(B):
        valid = ~pad[b]
        if valid.sum() == 0: 
            continue
        rows = ctx_ids[b, valid]
        sims = (name_norm[rows] @ name_norm[rows].T)
        sem = sims >= semantic_tau
        idxv = torch.where(valid)[0]
        allow[b, idxv[:,None], idxv[None,:]] |= sem

    # category
    if include_category and cat is not None:
        for b in range(B):
            valid = ~pad[b]
            if valid.sum() == 0: 
                continue
            rows = ctx_ids[b, valid]
            same = (cat.gather(0, rows)[:,None] == cat.gather(0, rows)[None,:])
            idxv = torch.where(valid)[0]
            allow[b, idxv[:,None], idxv[None,:]] |= same

    # disallow pads
    for b in range(B):
        m = pad[b]
        allow[b, m, :] = False
        allow[b, :, m] = False

    # convert to additive bias mask: 0 for allowed, -inf for banned
    bias = torch.zeros(B, T, T, device=device)
    bias[~allow] = float("-inf")
    return bias

# -------------------------
# Graph Transformer (custom layer to accept [B,T,T] attention bias)
# -------------------------

class GraphTransformerLayer(nn.Module):
    def __init__(self, d=512, h=8, drop=0.1):
        super().__init__()
        self.h = h; self.dk = d // h
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(drop); self.proj_drop = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(drop), nn.Linear(4*d, d))

    def forward(self, x, bias):  # x:[B,T,D], bias:[B,T,T] (0 or -inf)
        B,T,D = x.shape; H = self.h; dk = self.dk
        residual = x
        x = self.norm1(x)
        q = self.q(x).view(B,T,H,dk).transpose(1,2)   # [B,H,T,dk]
        k = self.k(x).view(B,T,H,dk).transpose(1,2)   # [B,H,T,dk]
        v = self.v(x).view(B,T,H,dk).transpose(1,2)   # [B,H,T,dk]
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(dk)  # [B,H,T,T]
        attn = attn + bias.unsqueeze(1)  # broadcast to heads
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B,H,T,dk]
        out = out.transpose(1,2).contiguous().view(B,T,D)
        out = self.proj_drop(self.out(out))
        x = residual + out

        # FFN
        x = x + self.ff(self.norm2(x))
        return x

class GraphUserEncoder(nn.Module):
    def __init__(self, dim=512, nhead=8, nlayers=2, dropout=0.1, edge_drop=0.1, max_len=64):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_len, dim))
        self.layers = nn.ModuleList([GraphTransformerLayer(dim, nhead, dropout) for _ in range(nlayers)])
        self.edge_drop = edge_drop
        self.norm = nn.LayerNorm(dim)
        self.pool_q = nn.Linear(dim, dim)
        self.pool = nn.MultiheadAttention(dim, nhead, batch_first=True)

    def forward(self, seq_emb, bias):
        # seq_emb: [B,T,D]; bias: [B,T,T] with 0 / -inf
        B,T,D = seq_emb.size()
        x = seq_emb + self.pos[:T].unsqueeze(0)
        # edge dropout
        if self.training and self.edge_drop > 0:
            keep = (torch.rand_like(bias) > self.edge_drop)
            # keep self edges
            eye = torch.eye(T, device=bias.device).bool().unsqueeze(0)
            keep = keep | eye
            bias = torch.where(keep, bias, torch.full_like(bias, float("-inf")))
        for layer in self.layers:
            x = layer(x, bias)
        x = self.norm(x)
        q = self.pool_q(x.mean(1, keepdim=True))
        u, _ = self.pool(q, x, x)   # [B,1,D]
        return u.squeeze(1)

# -------------------------
# Fusion module (learnable projection on concat(name, kg))
# -------------------------

class FusionModule(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Linear(2*dim, dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(dim)
    def forward(self, name_emb, kg_emb):
        x = torch.cat([name_emb, kg_emb], dim=-1)
        out = torch.tanh(self.proj(x))
        return self.norm(out)

# -------------------------
# Training & Evaluation
# -------------------------

@dataclass
class Args:
    ctx_len: int
    epochs: int
    lr: float
    negatives: int

def train_eval(train_dl, val_dl, test_dl,
               name_bank, kg_bank_raw, kg_proj, fusion,
               poi_meta, args, out_dir,
               spatial_km=1.0, semantic_tau=0.75, edge_drop=0.1,
               ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name_bank = F.normalize(name_bank.to(device), dim=-1)
    kg_emb = kg_proj(kg_bank_raw.to(device))
    kg_emb = F.normalize(kg_emb, dim=-1)

    # Build catalog fused embeddings lazily
    @torch.no_grad()
    def fuse_all() -> torch.Tensor:
        fusion.eval()
        B = 4096; N = name_bank.size(0); outs = []
        for i in range(0, N, B):
            n = name_bank[i:i+B]; k = kg_emb[i:i+B]
            f = F.normalize(fusion(n, k), dim=-1); outs.append(f)
        return torch.cat(outs, dim=0)

    # Prepare meta tensors for adjacency (aligned to POI index order)
    poi_meta = poi_meta.copy()
    poi_meta["venue_id"] = poi_meta["venue_id"].astype(str)
    # build aligned arrays
    N = name_bank.size(0)
    lat = torch.zeros(N, device=device); lon = torch.zeros(N, device=device)
    cat_map = {c:i for i,c in enumerate(sorted(poi_meta["venue_category"].fillna("UNK").unique()))}
    cat = torch.zeros(N, dtype=torch.long, device=device)
    # fill
    vid2row = {str(v):i for i,v in enumerate(poi_meta["venue_id"].tolist())}
    # Some rows may not match exact ordering; rely on name_bank order via saved poi_index externally:
    # Instead, we will accept that train/val dataset only uses valid indices, and we set lat/lon/cat by index pass:
    # To guarantee alignment, ask train_eval caller to pass a meta with "ix" column = row index. Build that now:
    if "ix" not in poi_meta.columns:
        # do nothing; we'll fill by matching 'ix' via external mapping
        pass

    # We ask caller to pass 'aligned_meta' with ix; to be robust, we just create empty tensors and fill on-demand:
    # For efficiency, construct per-index tensors by scanning a dict passed from main:
    # main will pass tensors lat_all, lon_all, cat_all aligned to poi_index ordering.
    # So here we just declare placeholders to satisfy type checking:
    # (we'll receive these as closures below from main)
    # encoders
    enc = GraphUserEncoder(dim=512, nhead=8, nlayers=2, dropout=0.1, edge_drop=edge_drop, max_len=args.ctx_len).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(kg_proj.parameters()) + list(fusion.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    ckpt = os.path.join(out_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best_a1 = -1.0

    # We’ll inject aligned tensors via closure set in main():
    global _LAT_ALIGNED, _LON_ALIGNED, _CAT_ALIGNED
    _LAT_ALIGNED = _LON_ALIGNED = _CAT_ALIGNED = None  # will be set in main()

    def fuse_rows(idx_1d):
        idx_1d = idx_1d.clamp(min=0, max=name_bank.size(0)-1)
        n = name_bank[idx_1d]; k = kg_emb[idx_1d]
        return F.normalize(fusion(n, k), dim=-1)

    def fuse_seq(ctx_ids):
        flat = ctx_ids.reshape(-1).clamp(min=0)
        fused_flat = fuse_rows(flat)
        return fused_flat.view(ctx_ids.size(0), ctx_ids.size(1), -1)

    def build_bias(ctx_ids, ctx_ts):
        # Uses global aligned lat/lon/cat tensors
        bias = build_adjacency(
            ctx_ids=ctx_ids,
            ctx_ts=ctx_ts,
            lat=_LAT_ALIGNED, lon=_LON_ALIGNED, cat=_CAT_ALIGNED,
            name_bank=name_bank,
            spatial_km=spatial_km, semantic_tau=semantic_tau,
            include_category=True,
            device=device
        )
        # also mask pads
        pad = (ctx_ids < 0)
        big_neg = torch.full((ctx_ids.size(0), ctx_ids.size(1), ctx_ids.size(1)), float("-inf"), device=device)
        for b in range(ctx_ids.size(0)):
            m = pad[b]
            big_neg[b, m, :] = float("-inf")
            big_neg[b, :, m] = float("-inf")
        bias = torch.maximum(bias, big_neg)
        return bias

    @torch.no_grad()
    def evaluate(dl, split="val"):
        enc.eval()
        catalog = fuse_all()
        ranks = []
        for b in dl:
            ctx_ids = b.ctx_ids.to(device); ctx_ts = b.ctx_ts.to(device); tgt = b.tgt_id.to(device)
            pad = (ctx_ids < 0)
            seq = fuse_seq(ctx_ids); seq[pad] = 0.0
            bias = build_bias(ctx_ids, ctx_ts)
            u = enc(seq, bias)                   # [B,D]
            scores = u @ catalog.T               # [B,N]
            order = scores.argsort(dim=1, descending=True)
            for i in range(scores.size(0)):
                r = (order[i] == tgt[i]).nonzero(as_tuple=False)
                ranks.append(int(r[0,0]) + 1 if r.numel() else catalog.size(0))
        a1 = accuracy_at_k(ranks,1); a5=accuracy_at_k(ranks,5); a10=accuracy_at_k(ranks,10); nd=ndcg_at_k(ranks,5)
        logging.info(f"[{split.upper()}] Acc@1={a1:.4f} Acc@5={a5:.4f} Acc@10={a10:.4f} NDCG@5={nd:.4f}")
        return a1

    N_all = name_bank.size(0)
    for ep in range(1, args.epochs+1):
        enc.train(); fusion.train(); kg_proj.train()
        run = 0.0
        for b in tqdm(train_dl, desc=f"Epoch {ep}"):
            ctx_ids = b.ctx_ids.to(device); ctx_ts = b.ctx_ts.to(device); tgt = b.tgt_id.to(device)
            pad = (ctx_ids < 0)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                seq = fuse_seq(ctx_ids); seq[pad] = 0.0
                bias = build_bias(ctx_ids, ctx_ts)
                u = enc(seq, bias)  # [B,512]

                K = args.negatives
                neg = torch.randint(0, N_all, (u.size(0), K), device=device)
                # avoid trivial collisions
                neg[neg == tgt.unsqueeze(1)] = (neg[neg == tgt.unsqueeze(1)] + 1) % N_all
                cand_idx = torch.cat([tgt.unsqueeze(1), neg], dim=1)  # [B,1+K]
                cand_flat = cand_idx.reshape(-1)
                cand = fuse_rows(cand_flat).view(u.size(0), K+1, -1)
                logits = (u.unsqueeze(1) * cand).sum(-1)  # [B,1+K]
                labels = torch.zeros(u.size(0), dtype=torch.long, device=device)
                loss = ce(logits, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += float(loss.detach().item())

        logging.info(f"Epoch {ep} train loss={run/len(train_dl):.4f}")
        a1 = evaluate(val_dl, split="val")
        if a1 > best_a1:
            best_a1 = a1
            torch.save({
                "enc": enc.state_dict(),
                "fusion": fusion.state_dict(),
                "kg_proj": kg_proj.state_dict(),
                "val_acc1": best_a1,
            }, ckpt)
            logging.info(f"Saved best checkpoint to {ckpt} (Val Acc@1={best_a1:.4f})")

    state = torch.load(ckpt, map_location=device)
    enc.load_state_dict(state["enc"]); fusion.load_state_dict(state["fusion"]); kg_proj.load_state_dict(state["kg_proj"])
    _ = evaluate(test_dl, split="test")

# -------------------------
# Main
# -------------------------

def main(
    data_root="data",
    city="new_york",
    kg_dir="runs/new_york/kg",
    runs_root="runs",
    ctx_len=24,
    epochs=25,
    batch_size=256,
    lr=3e-4,
    negatives=256,
    seed=42,
    num_workers=8,
    spatial_km=1.0,
    semantic_tau=0.75,
    edge_drop=0.1,
):
    set_seed(seed)
    run_city_dir = os.path.join(runs_root, city)
    os.makedirs(run_city_dir, exist_ok=True)
    setup_logger(os.path.join(run_city_dir, "training.log"))

    # Load splits
    train = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_train.csv"))
    valid = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_validation.csv"))
    test  = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_test.csv"))

    # Trajectories (δτ = 8h strict)
    train_traj = build_trajectories_delta(train, 8)
    valid_traj = build_trajectories_delta(valid, 8)
    test_traj  = build_trajectories_delta(test, 8)

    logging.info(f"Sample trajectories (first 3):")
    for uid, seq in train_traj[:3]:
        logging.info(f"uid={uid} len={len(seq)} names={[str(r.name) for r in seq]} times={[str(r.timestamp) for r in seq]}")

    # Name tower (precomputed)
    name_bank, poi_index, meta_old = load_name_bank(run_city_dir)
    logging.info(f"Name bank: {tuple(name_bank.shape)} POIs={len(poi_index)}")

    # Rebuild rich meta with lat/lon/category if needed
    poi_meta = rebuild_poi_meta_if_needed(meta_old, train, valid, test, run_city_dir)
    # Align meta to poi_index order
    poi_meta = poi_meta.copy()
    poi_meta["venue_id"] = poi_meta["venue_id"].astype(str)
    # Build aligned lat/lon/cat tensors
    N = len(poi_index)
    lat_all = torch.zeros(N); lon_all = torch.zeros(N)
    cat_all = torch.zeros(N, dtype=torch.long)
    cat_map = {c:i for i,c in enumerate(sorted(poi_meta["venue_category"].fillna("UNK").unique()))}
    for vid, ix in poi_index.items():
        row = poi_meta[poi_meta["venue_id"] == vid]
        if len(row):
            rr = row.iloc[-1]
            lat_all[ix] = float(rr.get("latitude", np.nan)) if pd.notna(rr.get("latitude", np.nan)) else 0.0
            lon_all[ix] = float(rr.get("longitude", np.nan)) if pd.notna(rr.get("longitude", np.nan)) else 0.0
            cat_all[ix] = cat_map.get(rr.get("venue_category","UNK"), 0)
        else:
            lat_all[ix] = 0.0; lon_all[ix] = 0.0; cat_all[ix] = 0

    # KG tower
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kg_raw, kg_proj = load_kg_embeddings_aligned(kg_dir, poi_index, target_dim=512, device=device)
    logging.info(f"KGE aligned: {tuple(kg_raw.shape)}")

    # Datasets & loaders
    train_ds = PairDataset(train_traj, poi_index, ctx_len=ctx_len)
    val_ds   = PairDataset(valid_traj, poi_index, ctx_len=ctx_len)
    test_ds  = PairDataset(test_traj,  poi_index, ctx_len=ctx_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)

    logging.info(f"Users={train['user_id'].nunique()}  Categories={len(cat_map)}")

    # Fusion
    fusion = FusionModule(dim=512).to(device)

    # Expose aligned tensors to train_eval via globals (keeps function signature simple)
    global _LAT_ALIGNED, _LON_ALIGNED, _CAT_ALIGNED
    _LAT_ALIGNED = lat_all.to(device)
    _LON_ALIGNED = lon_all.to(device)
    _CAT_ALIGNED = cat_all.to(device)

    args = Args(ctx_len=ctx_len, epochs=epochs, lr=lr, negatives=negatives)
    train_eval(
        train_dl, val_dl, test_dl,
        name_bank=name_bank, kg_bank_raw=kg_raw, kg_proj=kg_proj, fusion=fusion,
        poi_meta=poi_meta, args=args, out_dir=run_city_dir,
        spatial_km=spatial_km, semantic_tau=semantic_tau, edge_drop=edge_drop,
    )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--city", type=str, default="new_york")
    p.add_argument("--kg_dir", type=str, default="runs/new_york/kg")
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--ctx_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--negatives", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--spatial_km", type=float, default=1.0)
    p.add_argument("--semantic_tau", type=float, default=0.75)
    p.add_argument("--edge_drop", type=float, default=0.10)
    args = p.parse_args()
    main(**vars(args))
