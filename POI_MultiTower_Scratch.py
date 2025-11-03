#!/usr/bin/env python
# coding: utf-8

# # Multi‑Tower Next‑POI (Research‑Aligned, Simple & Modular)

# 
# **Key design** (non‑NN towers for Name & KG; only User tower is trainable):
# - Name Tower: precompute 512‑d ModernBERT embeddings for each POI name; saved + frozen.
# - Knowledge Tower: load pre‑trained KG embeddings (RotatE/MuRE…); frozen.
# - User Tower: spatio‑temporal Transformer over context; trained to predict next POI.
# - Fusion: simple fixed weighted sum (α·name + (1−α)·KG). No extra tower module.
# - Trajectories: strict δτ = 8 hours; last POI is the label.
# - Metrics: Acc@1/5/10, NDCG@5 on val/test. Logs + best checkpoint.
# 

# In[1]:


#get_ipython().system("pip install -q 'transformers>=4.44.0' accelerate safetensors")


# In[3]:


import os, sys, json, math, logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# In[4]:


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


# In[5]:


def build_trajectories_delta(df, delta_hours=8):
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


# In[6]:


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

    all = []
    for i in tqdm(range(0, len(texts), batch_size), desc="ModernBERT name embeddings"):
        batch = tok(texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        batch = {k: v.to(device) for k,v in batch.items()}
        out = enc(**batch)
        pooled = out.last_hidden_state.mean(dim=1)
        all.append(pooled.detach().cpu())
    name_emb = torch.cat(all, dim=0)

    if name_emb.size(1) != 512:
        torch.manual_seed(0)
        W = torch.randn(name_emb.size(1), 512)
        W = torch.linalg.qr(W).Q
        name_emb = name_emb @ W

    np.save(os.path.join(out_dir, "name_emb.npy"), name_emb.numpy())
    poi_ref[["venue_id","name","venue_city","venue_category","address"]].to_parquet(os.path.join(out_dir,"poi_meta.parquet"))
    json.dump({v:i for i,v in enumerate(vids)}, open(os.path.join(out_dir, "poi_index.json"), "w"))


# In[7]:


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
        torch.manual_seed(1)
        W = torch.randn(kg.size(1), target_dim)
        W = torch.linalg.qr(W).Q
        kg = kg @ W
    return kg.contiguous(), e2id


# In[8]:


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, trajectories, poi_index, fused_emb, ctx_len=16):
        self.items = []
        self.fused = fused_emb
        self.vid2row = poi_index
        self.ctx_len = ctx_len
        for uid, seq in trajectories:
            vids = [str(r.venue_id) for r in seq]
            if not all(v in self.vid2row for v in vids): continue
            ctx = vids[:-1]; tgt = vids[-1]
            self.items.append((uid, ctx, tgt))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        uid, ctx, tgt = self.items[idx]
        ids = [self.vid2row[v] for v in ctx][-self.ctx_len:]
        pad = [-1] * (self.ctx_len - len(ids))
        ids = pad + ids
        return {"uid": uid,
                "ctx_ids": torch.tensor(ids, dtype=torch.long),
                "tgt_id": torch.tensor(self.vid2row[tgt], dtype=torch.long)}
        
def collate(batch):
    return {
        "uid": [b["uid"] for b in batch],
        "ctx_ids": torch.stack([b["ctx_ids"] for b in batch], dim=0),
        "tgt_id": torch.stack([b["tgt_id"] for b in batch], dim=0),
    }


# In[9]:


class UserSTEncoder(nn.Module):
    def __init__(self, dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_len, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                           dim_feedforward=4*dim, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq_emb, pad_mask):
        x = seq_emb + self.pos[:seq_emb.size(1)].unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        w = (~pad_mask).float().unsqueeze(-1)
        return (h * w).sum(1) / w.sum(1).clamp(min=1e-6)


# In[10]:


def train_eval(train_dl, val_dl, test_dl, fused_emb, args, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fused = fused_emb.to(device)
    enc = UserSTEncoder(dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=args.ctx_len).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-2)
    ce  = nn.CrossEntropyLoss()

    ckpt = os.path.join(out_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best_a1 = -1.0

    def sample_negs(pos_ids, K):
        N = fused.size(0)
        return torch.randint(0, N-1, (pos_ids.size(0), K), device=device)

    @torch.no_grad()
    def evaluate(dl, split="val"):
        enc.eval(); ranks = []
        for b in dl:
            ctx = b["ctx_ids"].to(device)      # [B,T]
            tgt = b["tgt_id"].to(device)       # [B]
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)
            seq = fused[ctxc]; seq[pad] = 0.0  # [B,T,D]
            u = enc(seq, pad_mask=pad)         # [B,D]
            scores = u @ fused.T               # [B,N]
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
            tgt = b["tgt_id"].to(device, non_blocking=True)
            pad = (ctx < 0)
            ctxc = ctx.clamp(min=0)

            # Detach static fused embeddings so they don’t build a graph
            seq = fused[ctxc].detach()      
            seq[pad] = 0.0

            u = enc(seq, pad_mask=pad)      # [B, D]

            K = args.negatives
            neg = sample_negs(tgt, K)
            cand_idx = torch.cat([tgt.unsqueeze(1), neg], dim=1)

            cand = fused[cand_idx].detach() 

            logits = (u.unsqueeze(1) * cand).sum(-1)
            labels = torch.zeros(u.size(0), dtype=torch.long, device=device)
            loss = ce(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()                 
            opt.step()

            run += float(loss.item())

        logging.info(f"Epoch {ep} train loss={run/len(train_dl):.4f}")
        a1 = evaluate(val_dl, split="val")
        if a1 > best_a1:
            best_a1 = a1; torch.save(enc.state_dict(), ckpt)
            logging.info(f"Saved best checkpoint to {ckpt} (Val Acc@1={best_a1:.4f})")

    enc.load_state_dict(torch.load(ckpt, map_location=device))
    _ = evaluate(test_dl, split="test")


# In[11]:


def main(
    data_root="data",
    city="new_york",
    kg_dir="runs/new_york/kg",
    runs_root="runs",
    name_model="nomic-ai/modernbert-embed-base",
    ctx_len=16,
    epochs=8,
    batch_size=256,
    lr=1e-4,
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

    # Load splits
    train = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_train.csv"))
    valid = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_validation.csv"))
    test  = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_test.csv"))

    # Build trajectories with δτ=8h
    train_traj = build_trajectories_delta(train, delta_hours=8)
    valid_traj = build_trajectories_delta(valid, delta_hours=8)
    test_traj  = build_trajectories_delta(test,  delta_hours=8)

    logging.info("Sample trajectories (first 3):")
    for uid, seq in train_traj[:3]:
        logging.info(f"uid={uid} len={len(seq)} names={[str(r.name) for r in seq]} times={[str(r.timestamp) for r in seq]}")

    # Precompute Name Tower embeddings (once)
    name_dir = os.path.join(out_dir, "name_emb")
    if not os.path.exists(os.path.join(name_dir, "name_emb.npy")):
        logging.info("Precomputing ModernBERT name embeddings…")
        all_df = pd.concat([train, valid, test], ignore_index=True)
        precompute_name_embeddings(all_df, name_dir, model_id=name_model, max_len=max_len)
    name_emb = torch.from_numpy(np.load(os.path.join(name_dir, "name_emb.npy"))).float()
    poi_index = json.load(open(os.path.join(name_dir, "poi_index.json")))
    poi_meta  = pd.read_parquet(os.path.join(name_dir, "poi_meta.parquet"))
    logging.info(f"name_emb={tuple(name_emb.shape)} POIs={len(poi_index)}")

    # Load KG embeddings
    kg_emb, entity2id = load_kg_embeddings(kg_dir, target_dim=512)
    logging.info(f"kg_emb={tuple(kg_emb.shape)} entities={len(entity2id)}")

    # Fuse (non-NN): alpha*name + (1-alpha)*kg using POI::<venue_id> keys
    alpha = 0.5
    N = len(poi_index)
    fused = torch.zeros(N, 512)
    missing = 0
    
    if kg_emb.size(1) != name_emb.size(1):
        W = torch.nn.Linear(kg_emb.size(1), name_emb.size(1), bias=False)
        kg_emb = W(kg_emb)
    
    for _, row in poi_meta.iterrows():
        vid = str(row["venue_id"]); i = poi_index[vid]
        j = entity2id.get(f"POI::{vid}", None)
        ne = name_emb[i]
        if j is not None and j < kg_emb.size(0):
            fused[i] = alpha*ne + (1-alpha)*kg_emb[j]
        else:
            fused[i] = ne; missing += 1
    logging.info(f"Fused ready: {tuple(fused.shape)} (KG missing {missing}/{N})")

    if precompute_only:
        logging.info("Precompute-only run completed."); return

    # Datasets
    train_ds = PairDataset(train_traj, poi_index, fused, ctx_len=ctx_len)
    val_ds   = PairDataset(valid_traj, poi_index, fused, ctx_len=ctx_len)
    test_ds  = PairDataset(test_traj,  poi_index, fused, ctx_len=ctx_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    # Train & Eval (User Tower only)
    args = type("Args", (), dict(ctx_len=ctx_len, epochs=epochs, lr=lr, negatives=negatives))()
    train_eval(train_dl, val_dl, test_dl, fused, args, out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--city", type=str, default="new_york")
    p.add_argument("--kg_dir", type=str, default="runs/new_york/kg")
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--name_model", type=str, default="nomic-ai/modernbert-embed-base")
    p.add_argument("--ctx_len", type=int, default=16)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--negatives", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precompute_only", action="store_true")
    args = p.parse_args()
    main(**vars(args))

