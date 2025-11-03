#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, math, logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger

def accuracy_at_k(ranks, k):
    r = np.asarray(ranks, dtype=np.int64)
    return float((r <= k).mean())

def ndcg_at_k(ranks, k):
    r = np.asarray(ranks, dtype=np.int64)
    return float(((r <= k).astype(np.float32) / np.log2(r + 1)).mean())

def set_seed(seed=42):
    import random; random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_trajectories_delta(df, delta_hours=8):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    delta = timedelta(hours=delta_hours)
    trajs = []
    for uid, g in df.groupby("user_id", sort=False):
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

@torch.no_grad()
def precompute_name_embeddings(poi_df, out_dir, model_id='nomic-ai/modernbert-embed-base',
                               batch_size=2048, max_len=32, device=None):
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    poi_ref = poi_df.sort_values("timestamp").drop_duplicates("venue_id", keep="last")
    texts = poi_ref["name"].fillna("").astype(str).tolist()
    vids  = poi_ref["venue_id"].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    enc = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    all_chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="ModernBERT name embeddings"):
        batch = tok(texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        batch = {k: v.to(device) for k,v in batch.items()}
        out = enc(**batch)
        pooled = out.last_hidden_state.mean(dim=1)
        all_chunks.append(pooled.detach().cpu())
    name_emb = torch.cat(all_chunks, dim=0)
    if name_emb.size(1) != 512:
        torch.manual_seed(0)
        W = torch.randn(name_emb.size(1), 512)
        W = torch.linalg.qr(W).Q
        name_emb = name_emb @ W
    np.save(os.path.join(out_dir, "name_emb.npy"), name_emb.numpy())
    poi_ref[["venue_id","name","venue_city","venue_category","address"]].to_parquet(os.path.join(out_dir,"poi_meta.parquet"))
    json.dump({v:i for i,v in enumerate(vids)}, open(os.path.join(out_dir, "poi_index.json"), "w"))

def load_kg_embeddings(kg_dir, target_dim=512):
    e2id = json.load(open(os.path.join(kg_dir, "entity2id.json")))
    pt = os.path.join(kg_dir, "entity_embeddings.pt")
    npy = os.path.join(kg_dir, "entity_embeddings.npy")
    if os.path.exists(pt):
        kg = torch.load(pt, map_location='cpu')
        if isinstance(kg, dict) and "emb" in kg: kg = kg["emb"]
        kg = torch.as_tensor(kg, dtype=torch.float32)
    elif os.path.exists(npy):
        kg = torch.from_numpy(np.load(npy)).float()
    else:
        raise FileNotFoundError("KG embeddings not found (.pt or .npy).")
    if kg.size(1) != target_dim:
        proj = nn.Linear(kg.size(1), target_dim, bias=False)
        with torch.no_grad():
            kg = proj(kg)
    return kg.contiguous(), e2id

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories, poi_index, ctx_len=16):
        self.items = []
        self.vid2row = poi_index
        self.ctx_len = ctx_len
        for uid, seq in trajectories:
            vids = [str(r.venue_id) for r in seq]
            times = [pd.to_datetime(r.timestamp) for r in seq]
            if not all(v in self.vid2row for v in vids):
                continue
            ctx_vids = vids[:-1]; tgt_vid = vids[-1]
            ctx_times = times[:-1]
            ctx_ids = [self.vid2row[v] for v in ctx_vids]
            tgt_id  = self.vid2row[tgt_vid]
            ctx_hours = [t.hour for t in ctx_times]
            ctx_wdays = [t.weekday() for t in ctx_times]
            ids = ctx_ids[-self.ctx_len:]; hrs = ctx_hours[-self.ctx_len:]; wds = ctx_wdays[-self.ctx_len:]
            pad_len = self.ctx_len - len(ids)
            ids = ([-1]*pad_len) + ids; hrs = ([0]*pad_len) + hrs; wds = ([0]*pad_len) + wds
            self.items.append({
                "uid": uid,
                "ctx_ids": torch.tensor(ids, dtype=torch.long),
                "ctx_hour": torch.tensor(hrs, dtype=torch.long),
                "ctx_wday": torch.tensor(wds, dtype=torch.long),
                "tgt_id": torch.tensor(tgt_id, dtype=torch.long),
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def collate(batch):
    return {
        "uid": [b["uid"] for b in batch],
        "ctx_ids": torch.stack([b["ctx_ids"] for b in batch], dim=0),
        "ctx_hour": torch.stack([b["ctx_hour"] for b in batch], dim=0),
        "ctx_wday": torch.stack([b["ctx_wday"] for b in batch], dim=0),
        "tgt_id": torch.stack([b["tgt_id"] for b in batch], dim=0),
    }

class UserSTEncoder(nn.Module):
    def __init__(self, dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=32):
        super().__init__()
        self.dim = dim
        self.hour_emb = nn.Embedding(24, 32)
        self.wday_emb = nn.Embedding(7, 16)
        self.proj = nn.Linear(512 + 32 + 16, dim)
        self.pos = nn.Parameter(torch.zeros(max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead,
            dim_feedforward=4*dim, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(dim)
    def forward(self, poi_seq_emb, hour_idx, wday_idx, pad_mask):
        he = self.hour_emb(hour_idx)
        we = self.wday_emb(wday_idx)
        x = torch.cat([poi_seq_emb, he, we], dim=-1)
        x = self.proj(x)
        x = x + self.pos[:x.size(1)].unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        w = (~pad_mask).float().unsqueeze(-1)
        return (h * w).sum(1) / w.sum(1).clamp(min=1e-6)

class FusionAttention(nn.Module):
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(2*dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, name_emb, kg_emb):
        x = torch.cat([name_emb, kg_emb], dim=-1)
        weights = self.attn(x)
        fused = weights[:, 0:1] * name_emb + weights[:, 1:2] * kg_emb
        return fused, weights

def accuracy_at_k(ranks, k):
    r = np.asarray(ranks, dtype=np.int64)
    return float((r <= k).mean())

def ndcg_at_k(ranks, k):
    r = np.asarray(ranks, dtype=np.int64)
    return float(((r <= k).astype(np.float32) / np.log2(r + 1)).mean())

def train_eval(train_dl, val_dl, test_dl, fused_emb, args, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused = nn.functional.normalize(fused_emb.to(device), dim=-1)
    enc = UserSTEncoder(dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=args.ctx_len).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-2)
    ce  = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ckpt = os.path.join(out_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best_a1 = -1.0
    TEMPERATURE = 0.07
    def sample_semihard_negs(u_repr, K):
        with torch.no_grad():
            sims = u_repr @ fused.T
            probs = torch.softmax(sims, dim=1)
            neg_idx = torch.multinomial(probs, K, replacement=False)
        return neg_idx
    @torch.no_grad()
    def evaluate(dl, split="val"):
        enc.eval(); ranks = []
        for b in dl:
            ctx = b["ctx_ids"].to(device, non_blocking=True)
            hrs = b["ctx_hour"].to(device, non_blocking=True)
            wds = b["ctx_wday"].to(device, non_blocking=True)
            tgt = b["tgt_id"].to(device, non_blocking=True)
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)
            seq = fused[ctxc].detach()
            seq[pad] = 0.0
            u = enc(seq, hrs, wds, pad_mask=pad)
            u = nn.functional.normalize(u, dim=-1)
            scores = (u @ fused.T) / TEMPERATURE
            order = scores.argsort(dim=1, descending=True)
            for i in range(scores.size(0)):
                r = (order[i] == tgt[i]).nonzero(as_tuple=False)
                rank = int(r[0,0]) + 1 if r.numel() else fused.size(0)
                ranks.append(rank)
        a1 = accuracy_at_k(ranks,1); a5=accuracy_at_k(ranks,5); a10=accuracy_at_k(ranks,10); nd=ndcg_at_k(ranks,5)
        logging.info(f"[{split.upper()}] Acc@1={a1:.4f} Acc@5={a5:.4f} Acc@10={a10:.4f} NDCG@5={nd:.4f}")
        return a1
    for ep in range(1, args.epochs+1):
        enc.train(); run=0.0
        for b in tqdm(train_dl, desc=f"Epoch {ep}"):
            ctx = b["ctx_ids"].to(device, non_blocking=True)
            hrs = b["ctx_hour"].to(device, non_blocking=True)
            wds = b["ctx_wday"].to(device, non_blocking=True)
            tgt = b["tgt_id"].to(device, non_blocking=True)
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)
            seq = fused[ctxc].detach()
            seq[pad] = 0.0
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                u = enc(seq, hrs, wds, pad_mask=pad)
                u = nn.functional.normalize(u, dim=-1)
                K = args.negatives
                neg = sample_semihard_negs(u, K)
                cand_idx = torch.cat([tgt.unsqueeze(1), neg], dim=1)
                cand = fused[cand_idx].detach()
                cand = nn.functional.normalize(cand, dim=-1)
                logits = (u.unsqueeze(1) * cand).sum(-1) / TEMPERATURE
                labels = torch.zeros(u.size(0), dtype=torch.long, device=device)
                loss = ce(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += float(loss.item())
        logging.info(f"Epoch {ep} train loss={run/len(train_dl):.4f}")
        a1 = evaluate(val_dl, split="val")
        if a1 > best_a1:
            best_a1 = a1; torch.save(enc.state_dict(), ckpt)
            logging.info(f"Saved best checkpoint to {ckpt} (Val Acc@1={best_a1:.4f})")
    enc.load_state_dict(torch.load(ckpt, map_location=device))
    _ = evaluate(test_dl, split="test")

def main(
    data_root="data",
    city="new_york",
    kg_dir="runs/new_york/kg",
    runs_root="runs",
    name_model="nomic-ai/modernbert-embed-base",
    ctx_len=16,
    epochs=12,
    batch_size=256,
    lr=5e-5,
    max_len=32,
    negatives=256,
    seed=42,
    precompute_only=False,
):
    set_seed(seed)
    out_dir = os.path.join(runs_root, city)
    setup_logger(os.path.join(out_dir, "training.log"))
    logging.info("=== Config ===")
    logging.info(f"city={city}, ctx_len={ctx_len}, epochs={epochs}, batch_size={batch_size}, lr={lr}, max_len={max_len}")
    train = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_train.csv"))
    valid = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_validation.csv"))
    test  = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_test.csv"))
    train_traj = build_trajectories_delta(train, delta_hours=8)
    valid_traj = build_trajectories_delta(valid, delta_hours=8)
    test_traj  = build_trajectories_delta(test,  delta_hours=8)
    logging.info("Sample trajectories (first 3):")
    for uid, seq in train_traj[:3]:
        logging.info(f"uid={uid} len={len(seq)} names={[str(r.name) for r in seq]} times={[str(r.timestamp) for r in seq]}")
    name_dir = os.path.join(out_dir, "name_emb")
    if not os.path.exists(os.path.join(name_dir, "name_emb.npy")):
        logging.info("Precomputing ModernBERT name embeddings…")
        all_df = pd.concat([train, valid, test], ignore_index=True)
        precompute_name_embeddings(all_df, name_dir, model_id=name_model, max_len=max_len)
    name_emb = torch.from_numpy(np.load(os.path.join(name_dir, "name_emb.npy"))).float()
    poi_index = json.load(open(os.path.join(name_dir, "poi_index.json")))
    poi_meta  = pd.read_parquet(os.path.join(name_dir, "poi_meta.parquet"))
    logging.info(f"name_emb={tuple(name_emb.shape)} POIs={len(poi_index)}")
    kg_emb, entity2id = load_kg_embeddings(kg_dir, target_dim=512)
    logging.info(f"kg_emb={tuple(kg_emb.shape)} entities={len(entity2id)}")
    name_emb = nn.functional.normalize(name_emb, dim=-1)
    kg_emb = nn.functional.normalize(kg_emb, dim=-1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion = FusionAttention(dim=512).to(device)
    fusion.train()
    N = len(poi_index)
    n512 = torch.zeros(N, 512)
    k512 = torch.zeros(N, 512)
    missing = 0
    for _, row in poi_meta.iterrows():
        vid = str(row["venue_id"]); i = poi_index[vid]
        j = entity2id.get(f"POI::{vid}", None)
        n512[i] = name_emb[i]
        if j is not None and j < kg_emb.size(0):
            k512[i] = kg_emb[j]
        else:
            k512[i] = name_emb[i]; missing += 1
    logging.info(f"KG aligned for {N-missing}/{N} POIs (fallback=Name for missing).")
    with torch.no_grad():
        fused, attn_w = fusion(n512.to(device), k512.to(device))
    fused = fused.cpu()
    if precompute_only:
        logging.info("Precompute-only run completed."); return
    train_ds = PairDataset(train_traj, poi_index, ctx_len=ctx_len)
    val_ds   = PairDataset(valid_traj, poi_index, ctx_len=ctx_len)
    test_ds  = PairDataset(test_traj,  poi_index, ctx_len=ctx_len)
    num_workers = max(4, (os.cpu_count() or 8)//2)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    args_obj = type("Args", (), dict(ctx_len=ctx_len, epochs=epochs, lr=lr, negatives=negatives))()
    train_eval(train_dl, val_dl, test_dl, fused, args_obj, out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--city", type=str, default="new_york")
    p.add_argument("--kg_dir", type=str, default="runs/new_york/kg")
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--name_model", type=str, default="nomic-ai/modernbert-embed-base")
    p.add_argument("--ctx_len", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--negatives", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precompute_only", action="store_true")
    args = p.parse_args()
    main(**vars(args))
