#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, math, logging
from datetime import timedelta
from typing import List, Tuple, Dict

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
    return float(((r <= k).astype(np.float32)/np.log2(r+1)).mean()) if len(r) else 0.0

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
# Name embeddings (precomputed)
# -------------------------

def load_name_bank(run_city_dir: str):
    name_dir = os.path.join(run_city_dir, "name_emb")
    emb = torch.from_numpy(np.load(os.path.join(name_dir, "name_emb.npy"))).float()
    poi_index = json.load(open(os.path.join(name_dir, "poi_index.json")))
    meta = pd.read_parquet(os.path.join(name_dir, "poi_meta.parquet"))
    return emb, poi_index, meta

# -------------------------
# KG embeddings (align -> normalize -> mean-pad -> project)
# -------------------------

def load_kg_embeddings_with_projector(
    kg_dir: str,
    run_city_dir: str,
    target_dim: int = 512,
    device: str = "cpu",
):
    # entity embeddings file
    pt = os.path.join(kg_dir, "entity_embeddings.pt")
    npy = os.path.join(kg_dir, "entity_embeddings.npy")

    if os.path.exists(pt):
        kg = torch.load(pt, map_location="cpu")
        if isinstance(kg, dict) and "emb" in kg:
            kg = kg["emb"]
        kg = torch.as_tensor(kg, dtype=torch.float32)
    elif os.path.exists(npy):
        kg = torch.from_numpy(np.load(npy)).float()
    else:
        raise FileNotFoundError("No KG embeddings found (.pt or .npy).")

    # project to target_dim if needed (orthogonal)
    if kg.size(1) != target_dim:
        torch.manual_seed(7)
        W = torch.randn(kg.size(1), target_dim)
        W, _ = torch.linalg.qr(W)
        kg = kg @ W

    # normalize
    kg = F.normalize(kg, dim=-1)

    # align row count to POI set
    poi_index_path = os.path.join(run_city_dir, "name_emb", "poi_index.json")
    expected = None
    if os.path.exists(poi_index_path):
        with open(poi_index_path) as f:
            expected = len(json.load(f))
    if expected is not None and kg.size(0) != expected:
        if kg.size(0) < expected:
            mean_vec = kg.mean(0, keepdim=True)
            pad = mean_vec.repeat(expected - kg.size(0), 1)
            kg = torch.cat([kg, pad], dim=0)
            logging.info(f"[Align KG] Padded KG embeddings from {kg.size(0)-pad.size(0)} to {kg.size(0)} rows")
        else:
            kg = kg[:expected]
            logging.info(f"[Align KG] Trimmed KG embeddings to {expected} rows")

    # tiny projector to let KG re-scale / rotate if needed
    in_dim = kg.size(1)  # actual KG embedding dimension (e.g., 256)
    projector = nn.Linear(in_dim, target_dim, bias=True)
    with torch.no_grad():
        # Initialize to near-identity mapping for stable start
        eye_dim = min(in_dim, target_dim)
        projector.weight.zero_()
        projector.weight[:eye_dim, :eye_dim] = torch.eye(eye_dim)
        projector.bias.zero_()
    projector = projector.to(device)


    return kg.to(device), projector

# -------------------------
# Dataset
# -------------------------

class PairDataset(Dataset):
    def __init__(self, trajectories, poi_index, ctx_len=24):
        self.items = []
        self.vid2row = poi_index
        self.ctx_len = ctx_len
        for uid, seq in trajectories:
            vids = [str(getattr(r, "venue_id")) for r in seq]
            if not all(v in self.vid2row for v in vids):
                # drop any trajectory with a POI not in name bank (keeps indices valid)
                continue
            ctx = vids[:-1]; tgt = vids[-1]
            self.items.append((uid, ctx, tgt))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        uid, ctx, tgt = self.items[idx]
        ids = [self.vid2row[v] for v in ctx][-self.ctx_len:]
        pad = [-1] * (self.ctx_len - len(ids))
        ids = pad + ids
        return {
            "uid": uid,
            "ctx_ids": torch.tensor(ids, dtype=torch.long),
            "tgt_id": torch.tensor(self.vid2row[tgt], dtype=torch.long),
        }

def collate(batch):
    return {
        "uid": [b["uid"] for b in batch],
        "ctx_ids": torch.stack([b["ctx_ids"] for b in batch], dim=0),
        "tgt_id": torch.stack([b["tgt_id"] for b in batch], dim=0),
    }

# -------------------------
# Fusion (learnable projection, no elementwise product)
# -------------------------

class FusionModule(nn.Module):
    """
    Learnable fusion:
      concat([name, kg]) -> Linear(1024->512) -> tanh -> LayerNorm
    """
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Linear(2*dim, dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, name_emb, kg_emb):
        x = torch.cat([name_emb, kg_emb], dim=-1)
        out = torch.tanh(self.proj(x))
        return self.norm(out), None

# -------------------------
# User Spatio-Temporal Encoder (Transformer)
# -------------------------

class UserSTEncoder(nn.Module):
    def __init__(self, dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=64):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=4*dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq_emb, pad_mask):
        x = seq_emb + self.pos[:seq_emb.size(1)].unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        w = (~pad_mask).float().unsqueeze(-1)
        return (h*w).sum(1) / w.sum(1).clamp(min=1e-6)

# -------------------------
# Training & Evaluation
# -------------------------

def train_eval(train_dl, val_dl, test_dl, name_emb, kg_raw, kg_proj, fusion, args, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize name embeddings
    name_emb = F.normalize(name_emb.to(device), dim=-1)

    # Project + normalize KG
    with torch.no_grad():
        kg_emb = kg_proj(kg_raw)  # learnable projector (starts as identity)
        kg_emb = F.normalize(kg_emb, dim=-1)

    # Full catalog fused (computed lazily each eval or once and cached)
    def fuse_all():
        # compute in chunks to save memory
        B = 4096
        N = name_emb.size(0)
        outs = []
        fusion.eval()
        with torch.no_grad():
            for i in range(0, N, B):
                n = name_emb[i:i+B]
                k = kg_emb[i:i+B]
                f, _ = fusion(n, k)
                outs.append(F.normalize(f, dim=-1))
        return torch.cat(outs, dim=0)  # [N,512]

    # Model pieces
    enc = UserSTEncoder(dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=args.ctx_len).to(device)

    # IMPORTANT: optimize user encoder + fusion + kg projector
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(fusion.parameters()) + list(kg_proj.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    ce = nn.CrossEntropyLoss()

    ckpt = os.path.join(out_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best_a1 = -1.0

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # helpers
    def fuse_rows(idx_1d):
        # Safe clamp
        idx_1d = idx_1d.clamp(min=0, max=name_emb.size(0)-1)
        n = name_emb[idx_1d]
        k = kg_emb[idx_1d]
        f, _ = fusion(n, k)
        return F.normalize(f, dim=-1)

    def fuse_seq(ctx_ids, pad_mask):
        # ctx_ids: [B,T]
        flat = ctx_ids.reshape(-1)
        flat = flat.clamp(min=0, max=name_emb.size(0)-1)
        fused_flat = fuse_rows(flat)                    # [B*T, D]
        B, T = ctx_ids.size()
        fused = fused_flat.view(B, T, -1).clone()
        fused[pad_mask] = 0.0
        return fused

    def sample_negs(pos_ids, K, N):
        # Sample negatives uniformly (GPU)
        neg = torch.randint(0, N, (pos_ids.size(0), K), device=device)
        # ensure no collision with positives
        neg[neg == pos_ids.unsqueeze(1)] = (neg[neg == pos_ids.unsqueeze(1)] + 1) % N
        return neg

    @torch.no_grad()
    def evaluate(dl, split="val"):
        enc.eval()
        catalog = fuse_all()  # [N, D]
        ranks = []
        for b in dl:
            ctx = b["ctx_ids"].to(device)
            tgt = b["tgt_id"].to(device)
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)
            seq = fuse_seq(ctxc, pad_mask=pad)
            u = enc(seq, pad_mask=pad)                # [B, D]
            scores = u @ catalog.T                    # [B, N]
            order = scores.argsort(dim=1, descending=True)
            for i in range(scores.size(0)):
                r = (order[i] == tgt[i]).nonzero(as_tuple=False)
                rank = int(r[0,0]) + 1 if r.numel() else catalog.size(0)
                ranks.append(rank)
        a1 = accuracy_at_k(ranks,1); a5=accuracy_at_k(ranks,5); a10=accuracy_at_k(ranks,10); nd=ndcg_at_k(ranks,5)
        logging.info(f"[{split.upper()}] Acc@1={a1:.4f} Acc@5={a5:.4f} Acc@10={a10:.4f} NDCG@5={nd:.4f}")
        return a1

    # Train
    N = name_emb.size(0)
    for ep in range(1, args.epochs+1):
        enc.train(); fusion.train(); kg_proj.train()
        run = 0.0
        for b in tqdm(train_dl, desc=f"Epoch {ep}"):
            ctx = b["ctx_ids"].to(device)
            tgt = b["tgt_id"].to(device)
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                seq = fuse_seq(ctxc, pad_mask=pad)            # [B,T,D]
                u = enc(seq, pad_mask=pad)                    # [B,D]

                K = args.negatives
                neg = sample_negs(tgt, K, N)                  # [B,K]
                cand_idx = torch.cat([tgt.unsqueeze(1), neg], dim=1)  # [B,1+K]

                # build candidate fused embeddings
                cand_flat = cand_idx.reshape(-1)
                cand_fused = fuse_rows(cand_flat).view(u.size(0), K+1, -1)  # [B,1+K,D]

                logits = (u.unsqueeze(1) * cand_fused).sum(-1)  # [B,1+K]
                labels = torch.zeros(u.size(0), dtype=torch.long, device=device)  # positives at column 0
                loss = ce(logits, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

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

    # Test
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
    seed=42,
    num_workers=4,
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
    name_emb, poi_index, _ = load_name_bank(run_city_dir)
    logging.info(f"Name bank: {tuple(name_emb.shape)} POIs={len(poi_index)}")

    # KG tower
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kg_raw, kg_proj = load_kg_embeddings_with_projector(kg_dir, run_city_dir, target_dim=512, device=device)
    logging.info(f"KGE aligned: {tuple(kg_raw.shape)}")

    # Coverage stats
    # If you have entity2id.json, you can compute coverage precisely (optional)
    e2i_path = os.path.join(kg_dir, "entity2id.json")
    if os.path.exists(e2i_path):
        entity2id = json.load(open(e2i_path))
        overlap = sum(1 for v in poi_index if f"POI::{v}" in entity2id)
        logging.info(f"KGE entries present for ~{overlap}/{len(poi_index)} POIs (projected & padded).")

    # Datasets & loaders
    train_ds = PairDataset(train_traj, poi_index, ctx_len=ctx_len)
    val_ds   = PairDataset(valid_traj, poi_index, ctx_len=ctx_len)
    test_ds  = PairDataset(test_traj,  poi_index, ctx_len=ctx_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)

    logging.info(f"Users={train['user_id'].nunique()}")
    fusion = FusionModule(dim=512).to(device)

    class Args: pass
    args = Args()
    args.ctx_len=ctx_len; args.epochs=epochs; args.lr=lr; args.negatives=256

    train_eval(train_dl, val_dl, test_dl, name_emb, kg_raw, kg_proj, fusion, args, run_city_dir)

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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    main(**vars(args))
