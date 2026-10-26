"""Evaluation protocols.

``evaluate``      supervised protocol: rank of the true next place among *all* places of the
                  city (Acc@k, NDCG@k with the Massive-STEPS convention 1/log2(rank + 2)),
                  plus the return / exploration decomposition of Section 5 of the paper.
``evaluate_llm``  LLM-Move protocol of Massive-STEPS: 200 random test users x 1 trajectory,
                  candidates = true place + 100 places drawn uniformly at random; mean and
                  standard deviation over five draws.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .model import forward
from .utils import amp_dtype


def ranking_metrics(ranks, eval_ks=(1, 5, 10), ndcg_ks=(5, 10)):
    """ranks are 0-based (0 = the true place is ranked first)."""
    ranks = np.asarray(ranks)
    m = {f"acc@{k}": float((ranks < k).mean()) for k in eval_ks}
    for k in ndcg_ks:
        m[f"ndcg@{k}"] = float(np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0).mean())
    m["mrr"] = float((1.0 / (ranks + 1)).mean())
    return m


def _cond_rank(S, tgt, ok):
    """1-based rank of the target within one channel; 0 where the target is in the other one."""
    ar = torch.arange(S.size(0), device=S.device)
    st = S[ar, tgt]
    r = (S > st.unsqueeze(1)).sum(1) + 1
    return torch.where(ok & torch.isfinite(st), r, torch.zeros_like(r)).cpu().numpy()


@torch.no_grad()
def evaluate(enc, head, ci, qs, cfg, bs=None):
    enc.eval(); head.eval()
    bs = bs or cfg.ft_batch
    adt = amp_dtype(cfg, ci.device)
    ranks, ret, ya, pa = [], [], [], []
    cond = defaultdict(list)
    for s in range(0, len(qs), bs):
        out = forward(enc, head, ci, qs[s:s + bs], adt)
        m, logits = out["m"], out["logits"]
        tgt = m["tgt_pi"]
        ar = torch.arange(tgt.size(0), device=tgt.device)
        ranks.append((logits > logits.gather(1, tgt.unsqueeze(1))).sum(1).cpu().numpy())
        known = out["visited"][ar, tgt]
        ret.append(known.cpu().numpy())
        cond["explore"].append(_cond_rank(out["exp_cond"].float(), tgt, ~known))
        cond["return"].append(_cond_rank(out["ret_cond"].float(), tgt, known))
        a_t = m["tgt_act"].clamp(max=out["gate"].size(1) - 1)
        cond["pi"].append(torch.sigmoid(out["gate"][ar, a_t].float()).cpu().numpy())
        ya += m["tgt_act"].tolist(); pa += out["act_logits"].argmax(-1).tolist()
    ranks = np.concatenate(ranks) if ranks else np.zeros(0, int)
    ret = np.concatenate(ret) if ret else np.zeros(0, bool)
    met = ranking_metrics(ranks)
    for name, sel in (("return", ret), ("explore", ~ret)):        # Acc = rho A_R + (1-rho) A_E
        if sel.any():
            mm = ranking_metrics(ranks[sel])
            met[f"{name}_share"] = float(sel.mean())
            for k in ("acc@1", "acc@5", "ndcg@5"):
                met[f"{name}_{k}"] = mm[k]
    for name in ("explore", "return"):                            # rank within the right channel
        rk = np.concatenate(cond[name]); rk = rk[rk > 0]
        if rk.size:
            met[f"{name}_cond_acc@1"] = float((rk <= 1).mean())
            met[f"{name}_cond_acc@10"] = float((rk <= 10).mean())
    pis = np.concatenate(cond["pi"]) if cond["pi"] else np.zeros(0)
    if (~ret).any():
        met["gate_on_explore"] = float(pis[~ret].mean())
    if ret.any():
        met["gate_on_return"] = float(pis[ret].mean())
    met["next_activity_acc"] = float((np.asarray(ya) == np.asarray(pa)).mean()) if ya else float("nan")
    met.update(n_queries=int(len(ranks)), n_places=int(ci.P))
    met["_ranks"] = ranks.astype(np.int32)
    return met


@torch.no_grad()
def evaluate_llm(enc, head, ci, test_trails, cfg, n_users=200, seeds=(0, 1, 2, 3, 4)):
    enc.eval(); head.eval()
    adt = amp_dtype(cfg, ci.device)
    by_user = defaultdict(list)
    for ti in test_trails:
        if len(ci.trails[ti]["rows"]) >= 2:
            by_user[ci.trails[ti]["user_id"]].append(ti)
    users = sorted(by_user)
    res = []
    for sd in seeds:
        rng = np.random.RandomState(sd)
        g = torch.Generator(device="cpu").manual_seed(sd)
        sel = rng.choice(len(users), min(n_users, len(users)), replace=False)
        qs = []
        for ui in sel:
            ts = by_user[users[ui]]; ti = ts[rng.randint(len(ts))]
            qs.append((ti, len(ci.trails[ti]["rows"]) - 1))
        ranks = []
        for s in range(0, len(qs), 64):
            out = forward(enc, head, ci, qs[s:s + 64], adt)
            logits, tgt = out["logits"], out["m"]["tgt_pi"]
            B, K = tgt.size(0), min(100, ci.P - 1)
            cand = torch.stack([torch.randperm(ci.P, generator=g)[:K + 1] for _ in range(B)]).to(ci.device)
            has = (cand == tgt.unsqueeze(1)).any(1)
            cand[~has, -1] = tgt[~has]                       # true place + 100 random places
            sc = logits.gather(1, cand)
            pos = cand == tgt.unsqueeze(1)
            r = (sc > (sc * pos.float()).sum(1, keepdim=True)).sum(1)
            ranks.append(r.cpu().numpy())
        mt = ranking_metrics(np.concatenate(ranks))
        res.append([mt["acc@1"], mt["acc@5"], mt["ndcg@5"]])
    r = np.asarray(res)
    out = {"llm_draws": len(seeds)}
    for j, k in enumerate(("acc@1", "acc@5", "ndcg@5")):
        out[f"llm_{k}"] = float(r[:, j].mean()); out[f"llm_{k}_std"] = float(r[:, j].std())
    return out
