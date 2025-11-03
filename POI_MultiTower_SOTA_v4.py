#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, math, logging
from collections import defaultdict, Counter
from datetime import timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================
# Utils / Metrics / Logging
# =========================================================

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

# =========================================================
# Trajectories (δτ = 8 hours strict)
# =========================================================

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

# =========================================================
# Name bank (precomputed)
# =========================================================

def load_name_bank(run_city_dir: str):
    name_dir = os.path.join(run_city_dir, "name_emb")
    emb = torch.from_numpy(np.load(os.path.join(name_dir, "name_emb.npy"))).float()     # [N,512]
    poi_index = json.load(open(os.path.join(name_dir, "poi_index.json")))               # vid -> row
    meta = pd.read_parquet(os.path.join(name_dir, "poi_meta.parquet"))                  # has venue_category etc.
    return emb, poi_index, meta

# =========================================================
# KG bank (aligned by entity2id.json)
# =========================================================

def load_kg_embeddings_aligned(
    kg_dir: str,
    poi_index: Dict[str, int],
    target_dim: int = None,     # auto-detect if None
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
        raise FileNotFoundError("KG embeddings (.pt or .npy) not found")

    e2i_path = os.path.join(kg_dir, "entity2id.json")
    if not os.path.exists(e2i_path):
        raise FileNotFoundError("entity2id.json not found; required for alignment")
    entity2id = json.load(open(e2i_path))

    # --- auto-detect dimension ---
    dim_in = kg_raw.size(1)
    if target_dim is None:
        target_dim = dim_in
        logging.info(f"[KG] Detected embedding dim={dim_in}")

    # If mismatch, project orthogonally
    if dim_in != target_dim:
        torch.manual_seed(7)
        W = torch.randn(dim_in, target_dim)
        W, _ = torch.linalg.qr(W)
        kg_raw = kg_raw @ W
        logging.info(f"[KG] Projected from {dim_in}→{target_dim}")

    kg_raw = F.normalize(kg_raw, dim=-1)

    # align to poi_index order
    N = len(poi_index)
    kg_aligned = torch.empty(N, target_dim, dtype=torch.float32)
    mean_vec = kg_raw.mean(0)
    miss = 0
    for vid, i in poi_index.items():
        j = entity2id.get(f"POI::{vid}")
        if j is None:
            kg_aligned[i] = mean_vec; miss += 1
        else:
            kg_aligned[i] = kg_raw[j]
    logging.info(f"[KG Align] mapped {N-miss}/{N} POIs; filled {miss} with mean vector.")

    projector = nn.Linear(target_dim, target_dim, bias=True)
    nn.init.eye_(projector.weight); nn.init.zeros_(projector.bias)
    return kg_aligned.to(device), projector.to(device)


# =========================================================
# Side features: category/hour/weekday
# =========================================================

def build_category_index(poi_meta: pd.DataFrame) -> Tuple[Dict[str,int], Dict[str,int]]:
    cats = sorted(set(str(x) for x in poi_meta["venue_category"].fillna("UNK").values))
    cat2ix = {c: (i+1) for i, c in enumerate(cats)}   # reserve 0 for PAD
    # venue_id -> cat_id
    vid2cat = {}
    for r in poi_meta.itertuples(index=False):
        vid = str(r.venue_id)
        cat = str(r.venue_category) if pd.notna(r.venue_category) else "UNK"
        vid2cat[vid] = cat2ix.get(cat, 0)
    return cat2ix, vid2cat

# =========================================================
# Co-visit graph & popularity
# =========================================================

def build_covisit_and_pop(train_traj, poi_index, topk=50):
    # co-visit counts of next given current
    cov = defaultdict(Counter)
    pop = Counter()
    for _, seq in train_traj:
        vids = [str(getattr(r, "venue_id")) for r in seq if str(getattr(r, "venue_id")) in poi_index]
        for v in vids:
            pop[v] += 1
        for a, b in zip(vids[:-1], vids[1:]):
            cov[a][b] += 1
    # build hard candidate lists
    hard = {}
    for v, cnts in cov.items():
        # topk successors by count
        hard[v] = [x for x, _ in cnts.most_common(topk) if x in poi_index and x != v]
    # global popular list
    global_pop = [vid for vid, _ in pop.most_common(max(1000, topk))]
    return hard, pop, global_pop

# =========================================================
# Dataset with side features
# =========================================================

class PairDataset(Dataset):
    def __init__(self, trajectories, poi_index, vid2cat, ctx_len=24):
        self.items = []
        self.vid2row = poi_index
        self.vid2cat = vid2cat
        self.ctx_len = ctx_len
        for uid, seq in trajectories:
            vids = [str(getattr(r, "venue_id")) for r in seq]
            if not all(v in self.vid2row for v in vids): 
                continue
            ts   = [getattr(r, "timestamp") for r in seq]
            hrs  = [int(pd.Timestamp(t).hour) for t in ts]
            wds  = [int(pd.Timestamp(t).weekday()) for t in ts]
            cats = [self.vid2cat.get(v, 0) for v in vids]

            ctx_vids = vids[:-1]; tgt_vid = vids[-1]
            ctx_hrs  = hrs[:-1];  ctx_wds = wds[:-1]; ctx_cats = cats[:-1]
            self.items.append((uid, ctx_vids, ctx_cats, ctx_hrs, ctx_wds, tgt_vid))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        uid, vids, cats, hrs, wds, tgt_vid = self.items[idx]
        ids  = [self.vid2row[v] for v in vids][-self.ctx_len:]
        cids = cats[-self.ctx_len:]
        hids = hrs[-self.ctx_len:]
        wids = wds[-self.ctx_len:]

        # pad left
        pad_len = self.ctx_len - len(ids)
        pad = [-1]*pad_len
        ids  = pad + ids
        cids = [0]*pad_len + cids            # 0 = PAD
        hids = [-1]*pad_len + hids           # -1 will be turned to 0 by clamp
        wids = [-1]*pad_len + wids

        return {
            "uid": uid,
            "ctx_ids":  torch.tensor(ids,  dtype=torch.long),
            "ctx_cids": torch.tensor(cids, dtype=torch.long),
            "ctx_hrs":  torch.tensor(hids, dtype=torch.long),
            "ctx_wds":  torch.tensor(wids, dtype=torch.long),
            "tgt_id":   torch.tensor(self.vid2row[tgt_vid], dtype=torch.long),
        }

def collate(batch):
    return {
        "uid": [b["uid"] for b in batch],
        "ctx_ids":  torch.stack([b["ctx_ids"]  for b in batch], dim=0),
        "ctx_cids": torch.stack([b["ctx_cids"] for b in batch], dim=0),
        "ctx_hrs":  torch.stack([b["ctx_hrs"]  for b in batch], dim=0),
        "ctx_wds":  torch.stack([b["ctx_wds"]  for b in batch], dim=0),
        "tgt_id":   torch.stack([b["tgt_id"]   for b in batch], dim=0),
    }

# =========================================================
# Fusion (learnable)
# =========================================================

class FusionModule(nn.Module):
    """Adaptive fusion for arbitrary name/KG dimensions."""
    def __init__(self, name_dim=512, kg_dim=256, out_dim=512):
        super().__init__()
        self.name_dim = name_dim
        self.kg_dim = kg_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(name_dim + kg_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, name_emb, kg_emb):
        # If dims differ, auto-project smaller one
        if name_emb.size(-1) != self.name_dim or kg_emb.size(-1) != self.kg_dim:
            raise RuntimeError(
                f"Fusion input dims mismatch: name={name_emb.size(-1)}, kg={kg_emb.size(-1)}"
            )
        x = torch.cat([name_emb, kg_emb], dim=-1)
        out = torch.tanh(self.proj(x))
        return self.norm(out)

# =========================================================
# User Spatio-Temporal Encoder with side-feature adapters
# =========================================================

class UserSTEncoder(nn.Module):
    def __init__(self, dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=64,
                 n_cat=1, cat_dim=128, time_dim=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=4*dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(dim)

        # side embeddings
        self.cat_emb = nn.Embedding(n_cat+1, cat_dim, padding_idx=0)  # 0=PAD
        self.hour_emb = nn.Embedding(24+1, time_dim, padding_idx=0)   # 0=PAD; we store hour+1
        self.wday_emb = nn.Embedding(7+1,  time_dim, padding_idx=0)   # 0=PAD; weekday+1

        self.side_proj = nn.Linear(dim + cat_dim + time_dim + time_dim, dim)

    def forward(self, poi_seq, pad_mask, cat_ids, hour_ids, wday_ids):
        """
        poi_seq:  [B,T,512] fused POI embeddings
        cat_ids:  [B,T] (0..n_cat; 0 is PAD)
        hour_ids: [B,T] (-1..23) -> we add +1 and clamp to [0..24]
        wday_ids: [B,T] (-1..6)  -> +1 -> [0..7]
        """
        # build side features
        h_idx = torch.clamp(hour_ids + 1, min=0, max=24)
        w_idx = torch.clamp(wday_ids + 1,  min=0, max=7)

        c = self.cat_emb(cat_ids)
        h = self.hour_emb(h_idx)
        w = self.wday_emb(w_idx)

        x = torch.cat([poi_seq, c, h, w], dim=-1)
        x = self.side_proj(x)  # back to 512
        x = x + self.pos[:x.size(1)].unsqueeze(0)

        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        wmask = (~pad_mask).float().unsqueeze(-1)
        return (h * wmask).sum(1) / wmask.sum(1).clamp(min=1e-6)

# =========================================================
# Training & Evaluation (with hard negatives + popularity prior)
# =========================================================

def train_eval(train_dl, val_dl, test_dl,
               name_bank, kg_bank, kg_proj, fusion,
               args, out_dir,
               cat2ix, vid2cat, covis_hard, global_pop, pop_counter):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name_bank = F.normalize(name_bank.to(device), dim=-1)

    # Popularity prior (log count)
    N = name_bank.size(0)
    pop_arr = torch.zeros(N, dtype=torch.float32)
    for vid, idx in args.poi_index.items():
        pop_arr[idx] = math.log(1.0 + float(pop_counter.get(vid, 0)))
    pop_arr = pop_arr.to(device)

    temperature = 0.07

    # helpers -------------------------------------------------
    @torch.no_grad()
    def current_catalog():
        kg_now = F.normalize(kg_proj(kg_bank), dim=-1)
        fusion.eval()
        outs = []
        B = 4096
        for i in range(0, N, B):
            n = name_bank[i:i+B]; k = kg_now[i:i+B]
            f = fusion(n, k)
            outs.append(F.normalize(f, dim=-1))
        return torch.cat(outs, dim=0)  # [N,512]

    @torch.no_grad()
    def fuse_rows(idx_1d):
        idx_1d = idx_1d.clamp(min=0, max=N-1)
        n = name_bank[idx_1d]
        k = F.normalize(kg_proj(kg_bank)[idx_1d], dim=-1)
        f = fusion(n, k)
        return F.normalize(f, dim=-1)

    @torch.no_grad()
    def fuse_seq(ctx_ids, pad_mask):
        flat = ctx_ids.reshape(-1).clamp(min=0, max=N-1)
        fused_flat = fuse_rows(flat)                  # [B*T,512]
        B, T = ctx_ids.size()
        fused = fused_flat.view(B, T, -1).clone()
        fused[pad_mask] = 0.0
        return fused

    # hard negative bank -------------------------------------
    # per VID: precomputed list of hard negatives (co-visit successors + same-category populars)
    vid2hard = {}
    for vid, idx in args.poi_index.items():
        pool = []
        # co-visit hard
        pool.extend(covis_hard.get(vid, [])[:args.hard_topk_covisit])
        # same-category populars
        c = vid2cat.get(vid, 0)
        if c != 0:
            # take from global_pop but filter by same category
            same_cat = [v for v in global_pop if vid2cat.get(v, 0) == c and v != vid]
            pool.extend(same_cat[:args.hard_topk_samecat])
        # map to indices and dedup
        pool_idx = [args.poi_index[v] for v in pool if v in args.poi_index]
        vid2hard[idx] = torch.tensor(list(dict.fromkeys(pool_idx)), dtype=torch.long, device=device)

    def sample_negatives(tgt_ids, K):
        # Mix: α hard, (1-α) uniform
        alpha = args.hard_mix
        n_hard = int(K * alpha)
        n_uniform = K - n_hard
        B = tgt_ids.size(0)
        out = torch.empty(B, 0, dtype=torch.long, device=device)

        if n_hard > 0:
            hard_mat = torch.full((B, n_hard), -1, dtype=torch.long, device=device)
            for i in range(B):
                t = int(tgt_ids[i].item())
                pool = vid2hard.get(t, None)
                if pool is None or pool.numel() == 0:
                    continue
                if pool.numel() >= n_hard:
                    choice = pool[torch.randint(0, pool.numel(), (n_hard,), device=device)]
                else:
                    # sample with wrap
                    idx = torch.randint(0, pool.numel(), (n_hard,), device=device)
                    choice = pool[idx]
                hard_mat[i] = choice
            # fill any -1 with uniform fallback
            mask = (hard_mat < 0)
            if mask.any():
                hard_mat[mask] = torch.randint(0, N, (int(mask.sum().item()),), device=device)
            out = torch.cat([out, hard_mat], dim=1)

        if n_uniform > 0:
            uni = torch.randint(0, N, (B, n_uniform), device=device)
            out = torch.cat([out, uni], dim=1)

        # avoid positives
        out[out == tgt_ids.unsqueeze(1)] = (out[out == tgt_ids.unsqueeze(1)] + 1) % N
        return out  # [B,K]

    # model ---------------------------------------------------
    enc = UserSTEncoder(dim=512, nhead=8, nlayers=2, dropout=0.1, max_len=args.ctx_len,
                        n_cat=len(cat2ix), cat_dim=128, time_dim=32).to(device)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.05)

    ckpt = os.path.join(out_dir, "checkpoints", "best.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best_a1 = -1.0

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ---------- Stage 1: warm-up (freeze fusion + kg_proj) ----------
    for p in fusion.parameters(): p.requires_grad_(False)
    for p in kg_proj.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)

    @torch.no_grad()
    def evaluate(dl, split="val"):
        enc.eval()
        catalog = current_catalog()    # [N,512]
        ranks = []
        for b in dl:
            ctx = b["ctx_ids"].to(device)
            tgt = b["tgt_id"].to(device)
            cat = b["ctx_cids"].to(device)
            hrs = b["ctx_hrs"].to(device)
            wds = b["ctx_wds"].to(device)
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)
            seq = fuse_seq(ctxc, pad_mask=pad)
            u = enc(seq, pad_mask=pad, cat_ids=cat, hour_ids=hrs, wday_ids=wds)     # [B,512]
            scores = (u @ catalog.T) / temperature + args.pop_lambda * pop_arr      # [B,N]
            order = scores.argsort(dim=1, descending=True)
            for i in range(scores.size(0)):
                r = (order[i] == tgt[i]).nonzero(as_tuple=False)
                rank = int(r[0,0]) + 1 if r.numel() else catalog.size(0)
                ranks.append(rank)
        a1 = accuracy_at_k(ranks,1); a5=accuracy_at_k(ranks,5); a10=accuracy_at_k(ranks,10); nd=ndcg_at_k(ranks,5)
        logging.info(f"[{split.upper()}] Acc@1={a1:.4f} Acc@5={a5:.4f} Acc@10={a10:.4f} NDCG@5={nd:.4f}")
        return a1

    def train_one_epoch(opt):
        enc.train()
        fusion.eval(); kg_proj.eval()  # frozen in warmup
        run = 0.0
        catalog = current_catalog()    # refresh each epoch
        for b in tqdm(train_dl, desc="Epoch", leave=False):
            ctx = b["ctx_ids"].to(device)
            tgt = b["tgt_id"].to(device)
            cat = b["ctx_cids"].to(device)
            hrs = b["ctx_hrs"].to(device)
            wds = b["ctx_wds"].to(device)
            pad = (ctx < 0); ctxc = ctx.clamp(min=0)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                seq = fuse_seq(ctxc, pad_mask=pad)                                        # [B,T,512]
                u = enc(seq, pad_mask=pad, cat_ids=cat, hour_ids=hrs, wday_ids=wds)      # [B,512]

                # negatives (hard + uniform)
                K = args.negatives
                neg = sample_negatives(tgt, K)                                            # [B,K]
                cand_idx = torch.cat([tgt.unsqueeze(1), neg], dim=1)                      # [B,1+K]
                cand = catalog[cand_idx]                                                  # [B,1+K,512]

                logits = (u.unsqueeze(1) * cand).sum(-1) / temperature                    # [B,1+K]
                logits = logits + args.pop_lambda * pop_arr[cand_idx]                     # add prior
                labels = torch.zeros(u.size(0), dtype=torch.long, device=device)
                loss = ce(logits, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run += float(loss.detach().item())
        return run / max(1, len(train_dl))

    # warmup
    for ep in range(1, args.warmup_epochs+1):
        tr = train_one_epoch(opt)
        logging.info(f"[Warmup E{ep}] loss={tr:.4f}")
        _ = evaluate(val_dl, "val")

    # ---------- Stage 2: unfreeze (small LR) ----------
    for p in fusion.parameters(): p.requires_grad_(True)
    for p in kg_proj.parameters(): p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [{"params": enc.parameters(),    "lr": args.lr},
         {"params": fusion.parameters(), "lr": args.lr * 0.2},
         {"params": kg_proj.parameters(),"lr": args.lr * 0.1}],
        weight_decay=1e-4
    )
    remain = max(0, args.epochs - args.warmup_epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain) if remain > 0 else None

    # stage-2 epochs
    for ep in range(args.warmup_epochs+1, args.epochs+1):
        tr = train_one_epoch(opt)
        logging.info(f"Epoch {ep} train loss={tr:.4f}")
        a1 = evaluate(val_dl, "val")
        if a1 > best_a1:
            best_a1 = a1
            torch.save({
                "enc": enc.state_dict(),
                "fusion": fusion.state_dict(),
                "kg_proj": kg_proj.state_dict(),
                "val_acc1": best_a1,
            }, os.path.join(out_dir, "checkpoints", "best.pt"))
            logging.info(f"Saved best checkpoint (Val Acc@1={best_a1:.4f})")
        if sched is not None: sched.step()

    # Test
    state = torch.load(os.path.join(out_dir, "checkpoints", "best.pt"), map_location=device)
    enc.load_state_dict(state["enc"]); fusion.load_state_dict(state["fusion"]); kg_proj.load_state_dict(state["kg_proj"])
    _ = evaluate(test_dl, "test")

# =========================================================
# Main
# =========================================================

def main(
    data_root="data",
    city="new_york",
    kg_dir="runs/new_york/kg",
    runs_root="runs",
    ctx_len=24,
    epochs=30,
    warmup_epochs=3,
    batch_size=256,
    lr=3e-4,
    seed=42,
    num_workers=8,
    negatives=256,
    hard_topk_covisit=50,
    hard_topk_samecat=50,
    hard_mix=0.6,
    pop_lambda=0.05,
):
    set_seed(seed)
    run_city_dir = os.path.join(runs_root, city)
    os.makedirs(run_city_dir, exist_ok=True)
    setup_logger(os.path.join(run_city_dir, "training.log"))

    # splits
    train = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_train.csv"))
    valid = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_validation.csv"))
    test  = pd.read_csv(os.path.join(data_root, city, f"{city}_checkins_test.csv"))

    # trajectories (δτ = 8h)
    train_traj = build_trajectories_delta(train, 8)
    valid_traj = build_trajectories_delta(valid, 8)
    test_traj  = build_trajectories_delta(test, 8)

    logging.info("Sample trajectories (first 3):")
    for uid, seq in train_traj[:3]:
        logging.info(f"uid={uid} len={len(seq)} names={[str(r.name) for r in seq]} times={[str(r.timestamp) for r in seq]}")

    # name bank
    name_emb, poi_index, poi_meta = load_name_bank(run_city_dir)
    name_emb = F.normalize(name_emb, dim=-1)
    logging.info(f"Name bank: {tuple(name_emb.shape)} POIs={len(poi_index)}")

    # categories
    cat2ix, vid2cat = build_category_index(poi_meta)
    logging.info(f"Categories={len(cat2ix)}")

    # KG aligned
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kg_bank, kg_proj = load_kg_embeddings_aligned(kg_dir, poi_index, device=device)
    logging.info(f"KGE aligned: {tuple(kg_bank.shape)}")

    # co-visit & popularity
    covis_hard, pop_counter, global_pop = build_covisit_and_pop(train_traj, poi_index, topk=200)
    logging.info("Co-visit graph & popularity computed.")

    # datasets/loaders
    train_ds = PairDataset(train_traj, poi_index, vid2cat, ctx_len=ctx_len)
    val_ds   = PairDataset(valid_traj, poi_index, vid2cat, ctx_len=ctx_len)
    test_ds  = PairDataset(test_traj,  poi_index, vid2cat, ctx_len=ctx_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=False)

    logging.info(f"Users={train['user_id'].nunique()}")

    fusion = FusionModule(
        name_dim=name_emb.size(1),
        kg_dim=kg_bank.size(1),
        out_dim=512
    ).to(device)


    # pack args for train_eval
    class Args: pass
    args = Args()
    args.ctx_len = ctx_len
    args.epochs = epochs
    args.warmup_epochs = warmup_epochs
    args.lr = lr
    args.negatives = negatives
    args.hard_topk_covisit = hard_topk_covisit
    args.hard_topk_samecat = hard_topk_samecat
    args.hard_mix = hard_mix
    args.pop_lambda = pop_lambda
    args.poi_index = poi_index  # for popularity prior mapping

    train_eval(train_dl, val_dl, test_dl,
               name_emb, kg_bank, kg_proj, fusion,
               args, run_city_dir,
               cat2ix, vid2cat, covis_hard, global_pop, pop_counter)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--city", type=str, default="new_york")
    p.add_argument("--kg_dir", type=str, default="runs/new_york/kg")
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--ctx_len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--negatives", type=int, default=256)
    p.add_argument("--hard_topk_covisit", type=int, default=50)
    p.add_argument("--hard_topk_samecat", type=int, default=50)
    p.add_argument("--hard_mix", type=float, default=0.6)
    p.add_argument("--pop_lambda", type=float, default=0.05)
    args = p.parse_args()
    main(**vars(args))
