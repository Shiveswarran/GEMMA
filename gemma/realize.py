"""
gemma.realize -- GEMMA-R: history-conditioned REALIZATION + ADAPTATION (v5).

This is the "A" in GEMMA.  The pretrained backbone produces a city-agnostic
belief about the user's next move (latent z, hierarchy, semantic-ID code);
realization turns that belief into a ranking over ONE city's concrete POIs.

What changed vs v4, and why (each was a hard cap on accuracy):

  1. History in context.  The benchmark task is  f(current trajectory T',
     user history T_u) -> rank(P).  All test users appear in train, and the
     LLM zero-shot baselines get the user's 15 most recent train trajectories.
     v4 fed only the 2-5 check-in prefix.  v5 builds the SAME diary format the
     backbone was pretrained on:  [hist_1] SEP [hist_2] SEP ... SEP [prefix]
     and queries the predictor at the true next position.

  2. Soft hierarchy, no pruning.  v4 pruned to the top-3 predicted activities
     and gave rank=|V| to every target outside -> recall ceiling.  v5 uses the
     EXACT chain-rule path likelihood  log p(a_v) + log p(p_v|a_v) + log p(s_v|p_v)
     as one feature for every POI.

  3. Real adaptation capacity.  v4 trained an 8-input MLP on the last step of
     each trajectory, so full-shot == few-shot.  v5 fine-tunes the backbone
     (small LR) + a realization head on EVERY next-step pair:
        E_v = LN( W_n name_v + sum_l C_l[code_v,l] + A[a_v] + P[p_v] + S[s_v] ) + r_v
        q   = MLP([z, h_last, summ]) + u_user
        s_v = tau * cos(q, E_v)  +  sum_k w_k(q) * phi_k(v | context)
     r_v (per-POI) and u_user are ZERO-initialised: a POI/user without data
     falls back to its semantic-ID composition -> cold-start and few-shot safe
     (this is what semantic IDs buy over the random-ID tables of STHGCN/GETNext).
     w_k(q) are CONTEXT-GATED evidence weights (softplus => monotone evidence).

  4. Generate -> realize -> rerank.  The AR semantic-ID likelihood (GNPR-SID)
     is costly, so it is evaluated only on the top-m shortlist of the dense
     realization score (never a full-vocabulary AR scan):
        final_v = s_v + alpha * z( log p_AR(code_v | z) )      for v in top-m

  Evidence features phi_k (all dense over the city vocabulary, on GPU):
     hist    recency-weighted visits in the user's history trajectories
     prefix  visits in the current trajectory
     is_last same POI as the last check-in (unconstrained sign)
     ptrans  PERSONAL transitions last->v in the user's own history
     geo     -log(1 + km from the last check-in)
     hier    exact hierarchical path log-likelihood from the backbone
     covis   global transitions last->v            (target stats: few/full only)
     pop     log popularity                         (target stats: few/full only)
     hour    log P(hour | v), smoothed              (target stats: few/full only)
  Target-city statistics are computed LEAVE-ONE-TRAJECTORY-OUT for training
  queries, so the head never learns to trust a count that includes its own label.

Zero-shot = no target-city labels or statistics: the frozen (or source
post-trained) backbone + a realization head trained on the SOURCE cities; the
user's own history is inference-time context (same information LLM-Mob/LLM-ZS/
LLM-Move receive).
"""
from __future__ import annotations

import copy
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import TT_CHECKIN, TT_SEP
from .metrics import classification_metrics, ranking_metrics

FEATS = ["hist", "prefix", "is_last", "ptrans", "geo", "hier", "covis", "pop", "hour"]
STAT_FEATS = ("covis", "pop", "hour")
FI = {k: i for i, k in enumerate(FEATS)}
INIT_W = {"hist": 2.0, "prefix": 1.0, "is_last": 0.0, "ptrans": 1.0, "geo": 1.0,
          "hier": 0.3, "covis": 1.0, "pop": 0.2, "hour": 0.3}
UNSIGNED = {"is_last"}                       # all others are monotone evidence


def _inv_softplus(y):
    return math.log(math.expm1(max(y, 1e-4)))


# =========================================================================== #
#  City index: POI tensors, user histories, (leave-one-out) target statistics  #
# =========================================================================== #
class CityIndex:
    """Everything needed to build queries and dense features for one city.

    observed : trail indices whose check-ins are *known* (history context)
    stat     : trail indices used for target-city statistics, or None (zero-shot)
    """

    def __init__(self, city, observed, stat, cfg, device):
        self.city, self.cfg, self.device = city, cfg, device
        pt = city.poi_table
        self.P = len(pt["venue_ids"])
        id2row = pt["id2row"]
        self.row_pi = np.array([id2row[int(v)] for v in city.venue_id], np.int64)
        self.hour_row = city.df["timestamp"].dt.hour.values.astype(np.int64)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        self.lat = t(pt["lat"]); self.lon = t(pt["lon"])
        self.act = t(pt["act"], torch.long); self.pri = t(pt["pri"], torch.long)
        self.sec = t(pt["sec"], torch.long)
        self.name = t(pt["name_emb"])
        self.codes = None
        if city.codes is not None:
            v2c = {}
            for i in range(len(city.venue_id)):
                v2c.setdefault(int(city.venue_id[i]), city.codes[i])
            L = city.codes.shape[1]
            c = np.zeros((self.P, L), np.int64)
            for k, v in enumerate(pt["venue_ids"]):
                if int(v) in v2c:
                    c[k] = v2c[int(v)]
            self.codes = t(c, torch.long)

        self.trails = city.trails
        self.observed = set(int(i) for i in observed)
        self.user_trails = defaultdict(list)
        for i in sorted(self.observed, key=lambda i: self.trails[i]["t0"]):
            self.user_trails[self.trails[i]["user_id"]].append(i)
        users = sorted({tr["user_id"] for tr in self.trails})
        self.uid = {u: k for k, u in enumerate(users)}
        self.n_users = len(users)

        self.has_stats = stat is not None
        self.stat = set(int(i) for i in stat) if stat is not None else set()
        if self.has_stats:
            pop = np.zeros(self.P, np.float32)
            H = np.zeros((self.P, 24), np.float32)
            nxt = defaultdict(lambda: defaultdict(float))
            for i in self.stat:
                rows = self.trails[i]["rows"]; pis = self.row_pi[rows]
                np.add.at(pop, pis, 1.0)
                np.add.at(H, (pis, self.hour_row[rows]), 1.0)
                for a, b in zip(pis[:-1], pis[1:]):
                    nxt[int(a)][int(b)] += 1.0
            self.covis = {a: (np.fromiter(d.keys(), np.int64), np.fromiter(d.values(), np.float32))
                          for a, d in nxt.items()}
            self.pop = t(pop); self.H = t(H)
            g = H.sum(0) + 1.0
            self.hprior = t(g / g.sum())

    # ------------------------------------------------------------------ #
    def history(self, ti):
        """Observed trajectories of the query's user, excluding the query itself."""
        tr = self.trails[ti]
        if not getattr(self.cfg, "use_history", True):
            return []
        hs = [j for j in self.user_trails.get(tr["user_id"], []) if j != ti]
        if self.cfg.causal_history:
            hs = [j for j in hs if self.trails[j]["t0"] < tr["t0"]]
        return hs

    def queries(self, trail_ids, all_positions):
        """(trail_idx, j): predict rows[j] from rows[:j]."""
        out = []
        for ti in trail_ids:
            n = len(self.trails[ti]["rows"])
            if n < 2:
                continue
            if all_positions:
                out += [(ti, j) for j in range(1, n)]
            else:
                out.append((ti, n - 1))
        return out

    # ------------------------------------------------------------------ #
    def batch(self, qs, with_target=True):
        """Build backbone inputs + sparse evidence for a list of queries."""
        cfg, city = self.cfg, self.city
        B = len(qs)
        toks, meta = [], defaultdict(list)
        sp = {k: ([], [], []) for k in ("hist", "prefix", "ptrans", "c_cov", "c_pop", "c_hour")}
        for b, (ti, j) in enumerate(qs):
            tr = self.trails[ti]; rows = tr["rows"]
            prefix = rows[:j]; last = int(prefix[-1])
            last_pi = int(self.row_pi[last])
            hs = self.history(ti)
            # ---- context tokens: [recent history trails] SEP ... SEP [prefix]
            tk = []
            for h in hs[-cfg.hist_trails:]:
                tk.extend(self.trails[h]["rows"].tolist()); tk.append(-1)
            tk.extend(prefix.tolist())
            tk = tk[-cfg.hist_max_tokens:]
            while tk and tk[0] == -1:
                tk = tk[1:]
            toks.append(tk)
            # ---- personal evidence (ALL observed history, recency weighted)
            hs = hs[-200:]                                   # cap for heavy users
            nh = len(hs)
            for r_i, h in enumerate(hs):
                pis = self.row_pi[self.trails[h]["rows"]]
                w = 0.9 ** (nh - 1 - r_i)
                sp["hist"][0].extend([b] * len(pis)); sp["hist"][1].extend(pis.tolist())
                sp["hist"][2].extend([w] * len(pis))
                nx = np.where(pis[:-1] == last_pi)[0]
                if len(nx):
                    sp["ptrans"][0].extend([b] * len(nx)); sp["ptrans"][1].extend(pis[nx + 1].tolist())
                    sp["ptrans"][2].extend([1.0] * len(nx))
            ppis = self.row_pi[prefix]
            sp["prefix"][0].extend([b] * len(ppis)); sp["prefix"][1].extend(ppis.tolist())
            sp["prefix"][2].extend([1.0] * len(ppis))
            nx = np.where(ppis[:-1] == last_pi)[0]
            if len(nx):
                sp["ptrans"][0].extend([b] * len(nx)); sp["ptrans"][1].extend(ppis[nx + 1].tolist())
                sp["ptrans"][2].extend([1.0] * len(nx))
            # ---- leave-one-trajectory-out corrections of target statistics
            if self.has_stats and ti in self.stat:
                apis = self.row_pi[rows]; hrs = self.hour_row[rows]
                sp["c_pop"][0].extend([b] * len(apis)); sp["c_pop"][1].extend(apis.tolist())
                sp["c_pop"][2].extend([-1.0] * len(apis))
                qh = self.hour_row[last]
                m = hrs == qh
                sp["c_hour"][0].extend([b] * int(m.sum())); sp["c_hour"][1].extend(apis[m].tolist())
                sp["c_hour"][2].extend([-1.0] * int(m.sum()))
                nx = np.where(apis[:-1] == last_pi)[0]
                sp["c_cov"][0].extend([b] * len(nx)); sp["c_cov"][1].extend(apis[nx + 1].tolist())
                sp["c_cov"][2].extend([-1.0] * len(nx))
            meta["last_pi"].append(last_pi)
            meta["hour"].append(int(self.hour_row[last]))
            meta["user"].append(self.uid[tr["user_id"]])
            if with_target:
                tgt = int(rows[j])
                meta["tgt_pi"].append(int(self.row_pi[tgt]))
                meta["tgt_act"].append(int(city.act_id[tgt]))

        # ---- dense backbone tensors
        L = max(1, max(len(t) for t in toks))
        R = np.full((B, L), -2, np.int64)
        for b, tk in enumerate(toks):
            R[b, :len(tk)] = tk
        real = R >= 0; sep = R == -1
        Rc = np.clip(R, 0, None)
        cont = city.cont[Rc] * real[..., None]
        dv = self.device
        tt = lambda x: torch.as_tensor(x, device=dv)
        bb_in = {"cont": tt(cont.astype(np.float32)),
                 "act": tt(np.where(real, city.act_id[Rc], 0)),
                 "pri": tt(np.where(real, city.pri_id[Rc], 0)),
                 "sec": tt(np.where(real, city.sec_id[Rc], 0)),
                 "types": tt(np.where(sep, TT_SEP, TT_CHECKIN).astype(np.int64)),
                 "ctx_mask": tt((real | sep).astype(np.float32))}
        m = {k: tt(np.asarray(v, np.int64)) for k, v in meta.items()}
        m["sparse"] = {k: (tt(np.asarray(v[0], np.int64)), tt(np.asarray(v[1], np.int64)),
                           tt(np.asarray(v[2], np.float32))) for k, v in sp.items()}
        if with_target and self.codes is not None:
            m["tgt_code"] = self.codes[m["tgt_pi"]]
        m["B"] = B
        return bb_in, m

    # ------------------------------------------------------------------ #
    def dense(self, m, hier):
        """(B,P,F) evidence tensor.  hier:(B,P) path log-lik (may carry grad)."""
        B, P, dv = m["B"], self.P, self.device
        feats = torch.zeros(B, P, len(FEATS), device=dv)

        def scat(key):
            out = torch.zeros(B, P, device=dv)
            b, p, v = m["sparse"][key]
            if b.numel():
                out.index_put_((b, p), v, accumulate=True)
            return out
        feats[..., FI["hist"]] = torch.log1p(scat("hist"))
        feats[..., FI["prefix"]] = torch.log1p(scat("prefix"))
        feats[..., FI["ptrans"]] = torch.log1p(scat("ptrans"))
        ar = torch.arange(B, device=dv)
        feats[ar, m["last_pi"], FI["is_last"]] = 1.0
        la = self.lat[m["last_pi"]].unsqueeze(1); lo = self.lon[m["last_pi"]].unsqueeze(1)
        dx = (self.lon.unsqueeze(0) - lo) * 111.32 * torch.cos(torch.deg2rad(la))
        dy = (self.lat.unsqueeze(0) - la) * 111.32
        feats[..., FI["geo"]] = -torch.log1p(torch.sqrt(dx * dx + dy * dy))
        if self.has_stats:
            cov = torch.zeros(B, P, device=dv)
            bi, pi, vv = [], [], []
            for b, lp in enumerate(m["last_pi"].tolist()):
                if lp in self.covis:
                    ys, cs = self.covis[lp]
                    bi.append(np.full(len(ys), b)); pi.append(ys); vv.append(cs)
            if bi:
                cov.index_put_((torch.as_tensor(np.concatenate(bi), device=dv),
                                torch.as_tensor(np.concatenate(pi), device=dv)),
                               torch.as_tensor(np.concatenate(vv), device=dv), accumulate=True)
            cov = cov + scat("c_cov")
            pop = self.pop.unsqueeze(0) + scat("c_pop")
            hcol = self.H[:, m["hour"]].t() + scat("c_hour")
            alpha = 2.0
            ph = (hcol.clamp_min(0) + alpha * self.hprior[m["hour"]].unsqueeze(1)) / (pop.clamp_min(0) + alpha)
            feats[..., FI["covis"]] = torch.log1p(cov.clamp_min(0))
            feats[..., FI["pop"]] = torch.log1p(pop.clamp_min(0))
            feats[..., FI["hour"]] = torch.log(ph)
        feats = feats.clone()
        feats[..., FI["hier"]] = hier
        return feats


# =========================================================================== #
class RealizationHead(nn.Module):
    def __init__(self, d_model, name_dim, n_act, n_pri, n_sec, sid_levels, sid_K,
                 d_real=256):
        super().__init__()
        self.name_proj = nn.Linear(name_dim, d_real)
        self.code_emb = nn.ModuleList([nn.Embedding(sid_K, d_real) for _ in range(sid_levels)])
        self.a_emb = nn.Embedding(n_act, d_real)
        self.p_emb = nn.Embedding(n_pri, d_real)
        self.s_emb = nn.Embedding(n_sec, d_real)
        self.ln = nn.LayerNorm(d_real)
        self.q_mlp = nn.Sequential(nn.Linear(3 * d_model, d_real), nn.GELU(),
                                   nn.Linear(d_real, d_real))
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))
        self.w0 = nn.Parameter(torch.tensor(
            [INIT_W[k] if k in UNSIGNED else _inv_softplus(INIT_W[k]) for k in FEATS]))
        self.gate = nn.Linear(d_real, len(FEATS))
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.alpha_ar = nn.Parameter(torch.tensor(0.5))
        self.signed = torch.tensor([k in UNSIGNED for k in FEATS])
        self.use_emb = True
        self.poi_res = None; self.user_res = None
        self.d_real = d_real

    def attach_city(self, n_pois, n_users, device):
        """Zero-initialised per-POI / per-user residuals (few/full only)."""
        self.poi_res = nn.Embedding(n_pois, self.d_real).to(device)
        self.user_res = nn.Embedding(n_users, self.d_real).to(device)
        nn.init.zeros_(self.poi_res.weight); nn.init.zeros_(self.user_res.weight)

    def detach_city(self):
        self.poi_res = None; self.user_res = None

    def poi_matrix(self, ci):
        e = self.name_proj(ci.name) + self.a_emb(ci.act) + self.p_emb(ci.pri) + self.s_emb(ci.sec)
        if ci.codes is not None:
            for l, emb in enumerate(self.code_emb):
                if l < ci.codes.size(1):
                    e = e + emb(ci.codes[:, l].clamp(max=emb.num_embeddings - 1))
        e = self.ln(e)
        if self.poi_res is not None:
            e = e + self.poi_res.weight
        return e

    def weights(self, q):
        w = self.w0 + self.gate(q)
        s = self.signed.to(q.device)
        return torch.where(s, w, F.softplus(w))

    def forward(self, z1, h_last, summ, feats, E, user=None, stat_on=True):
        q = self.q_mlp(torch.cat([z1, h_last, summ], -1).float())
        if self.user_res is not None and user is not None:
            q = q + self.user_res(user)
        w = self.weights(q)                                           # (B,F)
        if not stat_on:
            mask = torch.tensor([k not in STAT_FEATS for k in FEATS], device=q.device)
            w = w * mask
        logits = (feats * w.unsqueeze(1)).sum(-1)
        if self.use_emb:
            logits = logits + self.log_tau.exp() * (F.normalize(q, dim=-1) @ F.normalize(E, dim=-1).t())
        return logits, w


# =========================================================================== #
def _forward(bb, head, ci, qs, train, amp_dtype, rerank_m, with_target=True):
    bb_in, m = ci.batch(qs, with_target=with_target)
    dev = ci.device
    amp_on = dev.startswith("cuda") and amp_dtype is not None
    with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_on):
        z1, h_last, summ = bb.encode_query(bb_in)
    z1, h_last, summ = z1.float(), h_last.float(), summ.float()
    hier = bb.path_logprob(z1, ci.act, ci.pri, ci.sec)
    feats = ci.dense(m, hier)
    E = head.poi_matrix(ci)
    logits, w = head(z1, h_last, summ, feats, E, m["user"], stat_on=ci.has_stats)
    out = {"logits": logits, "z1": z1, "m": m, "w": w}
    # ---- stage 2: AR semantic-ID rerank on the top-m shortlist
    use_ar = (rerank_m > 0 and ci.codes is not None and getattr(bb, "leaf", None) is not None
              and hasattr(bb.leaf, "score_pairs"))
    if use_ar:
        M = min(rerank_m, ci.P)
        top = logits.detach().topk(M, dim=1).indices                      # (B,M)
        if train and with_target:
            tgt = m["tgt_pi"]
            has = (top == tgt.unsqueeze(1)).any(1)
            top[~has, -1] = tgt[~has]
        ar = bb.leaf.score_pairs(z1, ci.codes[top])                        # (B,M)
        arz = (ar - ar.mean(1, keepdim=True)) / (ar.std(1, keepdim=True) + 1e-6)
        base = logits.gather(1, top)
        out["top"] = top
        out["short"] = (base.detach() if train else base) + head.alpha_ar * arz
    act_logits = bb.act_head(z1)
    out["act_logits"] = act_logits
    return out


def _ranks(out):
    """Rank of the truth under dense->rerank scoring (no recall cap)."""
    logits = out["logits"].detach(); tgt = out["m"]["tgt_pi"]
    dense_rank = (logits > logits.gather(1, tgt.unsqueeze(1))).sum(1)
    if "top" not in out:
        return dense_rank.cpu().numpy()
    top, short = out["top"], out["short"].detach()
    pos = (top == tgt.unsqueeze(1))
    in_top = pos.any(1)
    ts = (short * pos.float()).sum(1)
    r_short = (short > ts.unsqueeze(1)).sum(1)
    return torch.where(in_top, r_short, dense_rank).cpu().numpy()


def _loss(bb, out, ci, cfg, head_only=False):
    m = out["m"]
    L = F.cross_entropy(out["logits"], m["tgt_pi"])
    if "short" in out:
        lab = (out["top"] == m["tgt_pi"].unsqueeze(1)).float().argmax(1)
        L = L + F.cross_entropy(out["short"], lab)
    L = L + 0.5 * F.cross_entropy(out["act_logits"], m["tgt_act"])
    if (not head_only and "tgt_code" in m and getattr(bb, "leaf", None) is not None
            and hasattr(bb.leaf, "score_pairs")):
        L = L + 0.5 * bb.leaf(out["z1"], m["tgt_code"])
    return L


# =========================================================================== #
@torch.no_grad()
def evaluate(bb, head, ci, qs, cfg, bs=None):
    bb.eval(); head.eval()
    bs = bs or cfg.ft_batch
    ranks, ya, pa, rev = [], [], [], []
    for s in range(0, len(qs), bs):
        out = _forward(bb, head, ci, qs[s:s + bs], False, _amp(cfg, ci.device), cfg.rerank_m)
        ranks.append(_ranks(out))
        m = out["m"]; seen = torch.zeros(m["B"], ci.P, dtype=torch.bool, device=ci.device)
        for key in ("hist", "prefix"):
            b_, p_, _ = m["sparse"][key]
            if b_.numel():
                seen[b_, p_] = True
        rev.append(seen.gather(1, m["tgt_pi"].unsqueeze(1)).squeeze(1).cpu().numpy())
        ya += out["m"]["tgt_act"].tolist(); pa += out["act_logits"].argmax(-1).tolist()
    ranks = np.concatenate(ranks) if ranks else np.zeros(0)
    rev = np.concatenate(rev) if rev else np.zeros(0, bool)
    met = ranking_metrics(ranks)
    # revisit (target in user history/prefix) vs explore (new POI for the user):
    # history heuristics score ~0 on 'explore'; this is where a foundation model must win
    for name, sel in (("revisit", rev), ("explore", ~rev)):
        if sel.any():
            mm = ranking_metrics(ranks[sel])
            met[f"{name}_frac"] = float(sel.mean())
            for k in ("acc@1", "acc@5", "ndcg@5"):
                met[f"{name}_{k}"] = mm[k]
    act = classification_metrics(ya, pa)
    met.update({"n_queries": int(len(ranks)), "n_candidates": int(ci.P),
                "act@1": act["accuracy"], "act_macro_f1": act["macro_f1"]})
    return met


@torch.no_grad()
def evaluate_llm_protocol(bb, head, ci, test_trails, cfg, n_users=200, seeds=(0, 1, 2, 3, 4)):
    """Massive-STEPS zero-shot LLM protocol: 200 random test users x 1 random
    trajectory, Acc@1/5 + NDCG@5, mean +- std over seeds (SE of one draw ~0.03).
    Three candidate sets are reported, because the LLM tables rank a SMALL
    candidate list, not the city vocabulary:
      near100    : 100 POIs nearest the last check-in (truth may be missing)
      gtnear100  : same, ground truth injected (replaces the farthest)
      gtrand100  : ground truth + 100 random POIs (original LLM-Move protocol)"""
    bb.eval(); head.eval()
    by_user = defaultdict(list)
    for ti in test_trails:
        if len(ci.trails[ti]["rows"]) >= 2:
            by_user[ci.trails[ti]["user_id"]].append(ti)
    users = sorted(by_user)
    variants = ("near100", "gtnear100", "gtrand100")
    res = {v: [] for v in variants}
    use_ar = ci.codes is not None and hasattr(getattr(bb, "leaf", None), "score_pairs")
    for sd in seeds:
        rng = np.random.RandomState(sd)
        g = torch.Generator(device="cpu").manual_seed(sd)
        sel = rng.choice(len(users), min(n_users, len(users)), replace=False)
        qs = []
        for ui in sel:
            ts = by_user[users[ui]]; ti = ts[rng.randint(len(ts))]
            qs.append((ti, len(ci.trails[ti]["rows"]) - 1))
        ranks = {v: [] for v in variants}
        for s in range(0, len(qs), 64):
            out = _forward(bb, head, ci, qs[s:s + 64], False, _amp(cfg, ci.device), 0)
            logits, m = out["logits"], out["m"]
            tgt = m["tgt_pi"]; B = tgt.size(0); K = min(100, ci.P - 1)
            la = ci.lat[m["last_pi"]].unsqueeze(1); lo = ci.lon[m["last_pi"]].unsqueeze(1)
            dx = (ci.lon.unsqueeze(0) - lo) * torch.cos(torch.deg2rad(la)); dy = ci.lat.unsqueeze(0) - la
            near = (dx * dx + dy * dy).topk(K, largest=False, dim=1).indices
            gtn = near.clone()
            miss = ~(gtn == tgt.unsqueeze(1)).any(1)
            gtn[miss, -1] = tgt[miss]
            rnd = torch.stack([torch.randperm(ci.P, generator=g)[:K + 1] for _ in range(B)]).to(ci.device)
            has = (rnd == tgt.unsqueeze(1)).any(1)
            rnd[~has, -1] = tgt[~has]
            for v, cand in zip(variants, (near, gtn, rnd)):
                sc = logits.gather(1, cand)
                if use_ar:
                    ar = bb.leaf.score_pairs(out["z1"], ci.codes[cand])
                    sc = sc + head.alpha_ar * (ar - ar.mean(1, keepdim=True)) / (ar.std(1, keepdim=True) + 1e-6)
                pos = cand == tgt.unsqueeze(1)
                ts = (sc * pos.float()).sum(1)
                r = (sc > ts.unsqueeze(1)).sum(1)
                r = torch.where(pos.any(1), r, torch.full_like(r, 10**6))
                ranks[v].append(r.cpu().numpy())
        for v in variants:
            mt = ranking_metrics(np.concatenate(ranks[v]))
            res[v].append([mt["acc@1"], mt["acc@5"], mt["ndcg@5"]])
    out = {"llm_seeds": len(seeds)}
    for v in variants:
        r = np.asarray(res[v])
        pre = "llm_" if v == "gtnear100" else f"llm_{v}_"          # headline = gtnear100
        for j, k in enumerate(("acc@1", "acc@5", "ndcg@5")):
            out[pre + k] = float(r[:, j].mean()); out[pre + k + "_std"] = float(r[:, j].std())
    return out


def _amp(cfg, device):
    if not (cfg.amp and device.startswith("cuda")):
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# =========================================================================== #
def finetune(bb, head, jobs, cfg, device, logger, epochs, lr_head, lr_bb,
             val=None, tag=""):
    """Train head (+ backbone when lr_bb>0) on one or more (CityIndex, queries).
    jobs: list of (ci, queries); round-robin over cities each step.
    val : (ci, queries) for early stopping on acc@1 (or None = fixed epochs)."""
    tune_bb = lr_bb > 0
    for p in bb.parameters():
        p.requires_grad_(False)
    bb_params = []
    if tune_bb:
        for name, p in bb.named_parameters():
            if name.startswith(("embed_t", "enc_t", "motif", "weighter", "query_tower",
                                "poi_tower", "geo_head", "temporal_head")):
                continue
            p.requires_grad_(True); bb_params.append(p)
    groups = [{"params": [p for p in head.parameters() if p.requires_grad], "lr": lr_head,
               "weight_decay": 1e-4}]
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_bb, "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups)
    amp_dtype = _amp(cfg, device)
    rng = np.random.RandomState(cfg.seed)
    steps_ep = sum(math.ceil(min(len(q), cfg.ft_max_queries) / cfg.ft_batch) for _, q in jobs)
    total = max(1, epochs * steps_ep); step = 0
    best, best_state, bad = -1.0, None, 0
    vals = None
    if val is not None:                  # epoch -1: the initialisation is a candidate
        vals = val if isinstance(val, list) else [val]
        vms = [evaluate(bb, head, vci, vq, cfg) for vci, vq in vals]
        best = float(np.mean([x["acc@1"] for x in vms]))
        best_state = (copy.deepcopy({k: v.cpu() for k, v in bb.state_dict().items()}),
                      copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()}))
        logger.info(f"[{tag}] init val acc@1={best:.4f}")
    for ep in range(epochs):
        bb.train() if tune_bb else bb.eval(); head.train()
        order = []
        for k, (ci, q) in enumerate(jobs):
            idx = rng.permutation(len(q))[:cfg.ft_max_queries]
            order += [(k, idx[s:s + cfg.ft_batch]) for s in range(0, len(idx), cfg.ft_batch)]
        rng.shuffle(order)
        t0 = time.time(); tot = 0.0; n = 0
        for k, bidx in order:
            ci, q = jobs[k]
            for g, base in zip(opt.param_groups, [lr_head, lr_bb]):
                g["lr"] = base * 0.5 * (1 + math.cos(math.pi * step / total))
            out = _forward(bb, head, ci, [q[i] for i in bidx], True, amp_dtype, cfg.rerank_m)
            loss = _loss(bb, out, ci, cfg, head_only=not tune_bb)
            if not torch.isfinite(loss):
                step += 1; continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step(); step += 1
            tot += float(loss.detach()); n += 1
        msg = f"[{tag}] ft epoch {ep} loss={tot / max(1, n):.4f} ({time.time() - t0:.0f}s)"
        if val is not None:
            vms = [evaluate(bb, head, vci, vq, cfg) for vci, vq in vals]
            vm = {k: float(np.mean([x[k] for x in vms])) for k in ("acc@1", "acc@5", "act@1")}
            msg += f" | val acc@1={vm['acc@1']:.4f} acc@5={vm['acc@5']:.4f} act@1={vm['act@1']:.3f}"
            if vm["acc@1"] > best:
                best, bad = vm["acc@1"], 0
                best_state = (copy.deepcopy({k: v.cpu() for k, v in bb.state_dict().items()}),
                              copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()}))
            else:
                bad += 1
        with torch.no_grad():
            w = head.weights(torch.zeros(1, head.d_real, device=device))[0].tolist()
        msg += " | w0: " + " ".join(f"{k}={v:.2f}" for k, v in zip(FEATS, w)) + \
               f" alpha_ar={head.alpha_ar.item():.2f} tau={head.log_tau.exp().item():.1f}"
        logger.info(msg)
        if val is not None and bad >= 3:
            logger.info(f"[{tag}] early stop at epoch {ep}")
            break
    if best_state is not None:
        bb.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        logger.info(f"[{tag}] restored best epoch (val acc@1={best:.4f})")
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.eval(); head.eval()
    return head


def new_head(bb, ci, cfg, device):
    return RealizationHead(cfg.d_model, ci.name.size(1), bb.n_act, bb.n_pri, bb.n_sec,
                           cfg.sid_levels if cfg.use_semantic_id else 0,
                           cfg.sid_codebook, cfg.d_real).to(device)
