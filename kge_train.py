#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math, numpy as np, torch
from tqdm import tqdm

# ---------------- dataset ----------------
class KGDataset(torch.utils.data.Dataset):
    def __init__(self, triples_path, entity2id, rel2id):
        self.e2id, self.r2id = entity2id, rel2id
        H, R, T = [], [], []
        with open(triples_path) as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                if h in self.e2id and t in self.e2id and r in self.r2id:
                    H.append(self.e2id[h]); R.append(self.r2id[r]); T.append(self.e2id[t])
        self.h = torch.tensor(H, dtype=torch.long)
        self.r = torch.tensor(R, dtype=torch.long)
        self.t = torch.tensor(T, dtype=torch.long)
    def __len__(self): return self.h.size(0)
    def __getitem__(self, i): return self.h[i], self.r[i], self.t[i]

# ---------------- models ----------------
class BaseKGE(torch.nn.Module):
    def __init__(self, nE, nR, dim):
        super().__init__()
        self.E = torch.nn.Embedding(nE, dim)
        self.R = torch.nn.Embedding(nR, dim)
        self.reset()
    def reset(self):
        torch.nn.init.uniform_(self.E.weight, -0.01, 0.01)
        torch.nn.init.uniform_(self.R.weight, -0.01, 0.01)

class TransE(BaseKGE):
    def __init__(self, nE, nR, dim, p=2):
        super().__init__(nE, nR, dim); self.p = p
    def score(self, h, r, t):
        return -torch.norm(self.E(h) + self.R(r) - self.E(t), p=self.p, dim=-1)

class ComplEx(BaseKGE):
    def __init__(self, nE, nR, dim):
        super().__init__(nE, nR, dim*2)  # real+imag
    def score(self, h, r, t):
        E, R = self.E.weight, self.R.weight
        hr, hi = E.weight[:,:E.size(1)//2], E.weight[:,E.size(1)//2:]
        rr, ri = R.weight[:,:R.size(1)//2], R.weight[:,R.size(1)//2:]
        tr, ti = E.weight[:,:E.size(1)//2], E.weight[:,E.size(1)//2:]
        # lookup
        hr, hi = hr[h], hi[h]; rr, ri = rr[r], ri[r]; tr, ti = tr[t], ti[t]
        # <h, r, conj(t)>
        return torch.sum(hr*rr*tr + hr*ri*ti + hi*rr*ti - hi*ri*tr, dim=-1)

class RotatE(BaseKGE):
    def __init__(self, nE, nR, dim, gamma=12.0):
        super().__init__(nE, nR, dim)
        self.gamma = torch.nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.mod = 1.0
    def score(self, h, r, t):
        # interpret E and R as complex with split halves
        e = self.E(h); rr = self.R(r); t = self.E(t)
        d = e.size(-1)//2
        eh, eh_i = e[...,:d], e[...,d:]
        er, er_i = rr[...,:d], rr[...,d:]
        et, et_i = t[...,:d], t[...,d:]
        # rotation: h ∘ r ≈ t
        # complex mul: (a+ib)(c+id) = (ac - bd) + i(ad+bc)
        real = eh*er - eh_i*er_i
        imag = eh*er_i + eh_i*er
        diff_r = real - et; diff_i = imag - et_i
        return self.gamma - torch.norm(torch.cat([diff_r, diff_i], dim=-1), p=2, dim=-1)

class MuRE(BaseKGE):
    """Euclidean variant with relation-specific diagonal scaling (approx. from MuRE/MuRP family)."""
    def __init__(self, nE, nR, dim):
        super().__init__(nE, nR, dim)
        self.A = torch.nn.Embedding(nR, dim)  # diag scaling per relation
        torch.nn.init.ones_(self.A.weight)
    def score(self, h, r, t):
        a = self.A(r)             # [B,D]
        return -torch.norm(a*self.E(h) + self.R(r) - self.E(t), p=2, dim=-1)

# ---------------- training ----------------
def train_kge(kg_dir, method="transe", dim=512, epochs=20, batch_size=4096, lr=5e-2,
              neg_ratio=1, loss_type="margin", margin=1.0, fp16=False, seed=42, save_every=0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    e2id = json.load(open(os.path.join(kg_dir,"entity2id.json")))
    r2id = json.load(open(os.path.join(kg_dir,"rel2id.json")))
    ds   = KGDataset(os.path.join(kg_dir,"triples.tsv"), e2id, r2id)
    dl   = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    nE, nR = len(e2id), len(r2id)
    method = method.lower()
    if method   == "transe":  model = TransE(nE, nR, dim)
    elif method == "complex": model = ComplEx(nE, nR, dim//2)  # internal doubles to 2*dim/2 = dim
    elif method == "rotate":  model = RotatE(nE, nR, dim)
    elif method == "mure":    model = MuRE(nE, nR, dim)
    else: raise ValueError("Unknown method")
    model = model.to(device)

    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    def batch_loss(h, r, t):
        B = h.size(0)
        # corrupt tails
        t_neg = torch.randint(0, nE, (B*neg_ratio,), device=device)
        h = h.to(device); r = r.to(device); t = t.to(device)
        with torch.cuda.amp.autocast(enabled=fp16):
            pos = model.score(h,r,t)
            h_rep = h.repeat_interleave(neg_ratio)
            r_rep = r.repeat_interleave(neg_ratio)
            pos_rep = pos.repeat_interleave(neg_ratio)
            neg = model.score(h_rep, r_rep, t_neg)
            if loss_type == "margin":
                loss = torch.relu(margin - pos_rep + neg).mean()
            else:  # InfoNCE over (1+neg_ratio)
                logits = torch.cat([pos.unsqueeze(1), neg.view(B,neg_ratio)], dim=1)
                labels = torch.zeros(B, dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    best = float("inf")
    for ep in range(1, epochs+1):
        model.train(); total = 0.0
        for h,r,t in tqdm(dl, desc=f"{method.upper()} epoch {ep}"):
            loss = batch_loss(h,r,t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += float(loss.item())
        avg = total / max(1, len(dl))
        print(f"[{method}] epoch {ep} loss={avg:.4f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(kg_dir, f"{method}_best.pt"))
            print("  saved:", os.path.join(kg_dir, f"{method}_best.pt"))
        if save_every and ep % save_every == 0:
            torch.save(model.state_dict(), os.path.join(kg_dir, f"{method}_ep{ep}.pt"))

    # Export entity embeddings for downstream fusion
    with torch.no_grad():
        E = model.E.weight.detach().cpu()
        torch.save(E, os.path.join(kg_dir, "entity_embeddings.pt"))
        print("Exported entity embeddings to:", os.path.join(kg_dir, "entity_embeddings.pt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg_dir", required=True)
    ap.add_argument("--method", choices=["transe","rotate","complex","mure"], default="transe")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--neg_ratio", type=int, default=1)
    ap.add_argument("--loss_type", choices=["margin","nce"], default="margin")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=0)
    args = ap.parse_args()
    train_kge(**vars(args))

if __name__ == "__main__":
    main()
