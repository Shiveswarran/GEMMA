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
import os
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
        ts = city.df["timestamp"]
        self.day_row = ((ts - ts.min()).dt.total_seconds().values / 86400.0).astype(np.float64)
        self.wkend_row = (ts.dt.dayofweek.values >= 5).astype(np.int64)
        self.act_row = np.asarray(city.act_id, np.int64)
        self.n_act = int(max(int(np.max(pt["act"])), int(np.max(self.act_row)))) + 1
        self._act_groups = None
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
        self.stats_off = False          # per-batch switch (stat dropout during source training)
        self.collab = None
        if self.has_stats and getattr(cfg, "collab", False):
            from .collab import CollabStats
            self.collab = CollabStats(self, sorted(self.stat), getattr(cfg, "collab_user_cap", 30),
                                      getattr(cfg, "collab_topk", 100))

    # ------------------------------------------------------------------ #
    def history(self, ti):
        """Observed trajectories of the query's user, excluding the query itself."""
        tr = self.trails[ti]
        if not getattr(self.cfg, "use_history", True):
            return []
        hs = [j for j in self.user_trails.get(tr["user_id"], []) if j != ti]
        if self.cfg.causal_history:
            hs = [j for j in hs if self.trails[j]["t0"] < tr["t0"]]
        kl = getattr(self.cfg, "hist_last", -1)
        if kl is not None and kl >= 0:    # LLM-protocol parity: only the user's last k observed
            hs = hs[-kl:] if kl > 0 else []   # trajectories (Massive-STEPS: 15), not causal
        k = getattr(self.cfg, "hist_budget", -1)
        if k is not None and k >= 0:      # cold start: the k most recent trajectories BEFORE the query
            hs = sorted((j for j in hs if self.trails[j]["t0"] < tr["t0"]),
                        key=lambda j: self.trails[j]["t0"])
            hs = hs[-k:] if k > 0 else []
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
        sp = {k: ([], [], []) for k in ("hist", "prefix", "ptrans", "c_cov", "c_pop", "c_hour",
                                        "tod", "dow", "rec", "icf")}
        last_new = np.zeros(B, np.float32); own_out = np.zeros(B, np.float32)
        use_collab = self.collab is not None and not self.stats_off
        A = self.n_act
        u_act = np.zeros((B, A), np.float32); u_atr = np.zeros((B, A), np.float32)
        u_new = np.zeros((B, A), np.float32); u_S = np.zeros(B, np.float32)
        u_pnew = np.zeros(B, np.float32); anchors = np.zeros((B, 5), np.int64)
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
            # ---- user-level behaviour statistics (return/explore + activity profile)
            qh, qw, qd = int(self.hour_row[last]), int(self.wkend_row[last]), float(self.day_row[last])
            la_ = int(self.act_row[last])
            seen, cnt, recd = set(), defaultdict(float), {}
            hrows = [self.trails[h]["rows"] for h in hs]
            for rws in hrows:
                pis = self.row_pi[rws]; acts = self.act_row[rws]
                for k in range(len(rws)):
                    v = int(pis[k]); a = int(acts[k]); r = int(rws[k])
                    u_act[b, a] += 1.0
                    if v not in seen:
                        u_new[b, a] += 1.0; seen.add(v)
                    cnt[v] += 1.0
                    dd = abs(qd - float(self.day_row[r]))
                    if dd < recd.get(v, 1e9):
                        recd[v] = dd
                    hd = abs(int(self.hour_row[r]) - qh); hd = min(hd, 24 - hd)
                    if hd <= 2:
                        sp["tod"][0].append(b); sp["tod"][1].append(v); sp["tod"][2].append(1.0)
                    if int(self.wkend_row[r]) == qw:
                        sp["dow"][0].append(b); sp["dow"][1].append(v); sp["dow"][2].append(1.0)
                for k in range(len(rws) - 1):
                    if int(acts[k]) == la_:
                        u_atr[b, int(acts[k + 1])] += 1.0
            for v, dd in recd.items():
                sp["rec"][0].append(b); sp["rec"][1].append(v); sp["rec"][2].append(float(np.exp(-dd / 14.0)))
            ppis = self.row_pi[prefix]
            pacts = self.act_row[prefix]
            for k in range(len(prefix) - 1):
                if int(pacts[k]) == la_:
                    u_atr[b, int(pacts[k + 1])] += 1.0
            u_S[b] = float(len(seen))
            last_new[b] = float(last_pi not in seen and last_pi not in set(ppis[:-1].tolist()))
            if use_collab:
                hw = defaultdict(float)
                for r_i, h in enumerate(hs):
                    for v in self.row_pi[self.trails[h]["rows"]]:
                        hw[int(v)] += 0.9 ** (nh - 1 - r_i)
                for v in ppis:
                    hw[int(v)] += 1.0
                ip, iv = self.collab.icf(hw, tr["user_id"])
                if len(ip):
                    sp["icf"][0].extend([b] * len(ip)); sp["icf"][1].extend(ip.tolist())
                    sp["icf"][2].extend(iv.tolist())
            u_pnew[b] = float(np.mean([int(v) not in seen for v in ppis])) if len(ppis) else 0.0
            top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
            anc = [v for v, _ in top] + [last_pi] * (5 - len(top))
            anchors[b] = anc
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
                own_out[b] = float(len(nx))
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
        m["u_act"] = tt(u_act); m["u_atr"] = tt(u_atr); m["u_new"] = tt(u_new)
        m["u_S"] = tt(u_S); m["u_pnew"] = tt(u_pnew); m["anchors"] = tt(anchors)
        m["last_new"] = tt(last_new); m["own_out"] = tt(own_out)
        m["wkend"] = tt(np.asarray([int(self.wkend_row[self.trails[ti]["rows"][j - 1]])
                                    for ti, j in qs], np.int64))
        tok_pi = np.where(real, self.row_pi[Rc], -1)
        m["tok_pi"] = tt(tok_pi.astype(np.int64))
        if with_target and self.codes is not None:
            m["tgt_code"] = self.codes[m["tgt_pi"]]
        m["B"] = B
        return bb_in, m

    def _cov(self, m, scat):
        """(B,P) population transition counts last->v, leave-one-trajectory-out."""
        B, P, dv = m["B"], self.P, self.device
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
        return cov + scat("c_cov")

    def collab_dense(self, m):
        """v7 population evidence (B,P): ctr, ccat, icf.  None if statistics are off."""
        if self.collab is None or self.stats_off or not self.has_stats:
            return None
        B, P, dv = m["B"], self.P, self.device

        def scat(key):
            out = torch.zeros(B, P, device=dv)
            b, p, v = m["sparse"][key]
            if b.numel():
                out.index_put_((b, p), v, accumulate=True)
            return out
        cs = self.collab
        if not hasattr(self, "_cl_t"):
            self._cl_t = {"logT": torch.as_tensor(cs.logT, device=dv),
                          "nsec": torch.as_tensor(cs.n_sec, device=dv),
                          "out": torch.as_tensor(cs.out, device=dv)}
        logT, nsec, outd = self._cl_t["logT"], self._cl_t["nsec"], self._cl_t["out"]
        ls = self.sec[m["last_pi"]]                                     # (B,)
        ccat = logT[ls][:, self.sec]                                    # (B,P) log T[sec(last), sec(v)]
        pcat = ccat.exp() / nsec[self.sec].clamp_min(1.0).unsqueeze(0)  # sums to 1 over POIs
        k = float(getattr(self.cfg, "ctr_kappa", 5.0))
        cov = self._cov(m, scat).clamp_min(0)
        den = (outd[m["last_pi"]] - m["own_out"]).clamp_min(0) + k     # (B,)
        ctr = torch.log((cov + k * pcat) / den.unsqueeze(1))
        return {"ctr": ctr, "ccat": ccat, "icf": scat("icf")}

    def act_groups(self):
        if self._act_groups is None:
            a = self.act.cpu().numpy()
            self._act_groups = [torch.as_tensor(np.where(a == k)[0], device=self.device)
                                for k in range(self.n_act)]
        return self._act_groups

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
        if self.has_stats and not self.stats_off:
            cov = self._cov(m, scat)
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
    if getattr(head, "kind", "") == "hrem":
        from .hrem import hrem_forward
        return hrem_forward(bb, head, ci, qs, train, amp_dtype, with_target)
    if getattr(head, "kind", "") == "wm":
        from .hsjepa import wm_forward
        return wm_forward(bb, head, ci, qs, train, amp_dtype, with_target)
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
        # NOTE (v5c): the truth is NOT forced into the shortlist during training.
        # v5b forced it in when missing; the forced items are low-dense-score
        # 'explore' POIs, so the rerank weight learned to reward LOW AR likelihood
        # (alpha_ar < 0) and then mis-reranked real shortlists at test time.
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
    if out.get("wm"):
        from .hrem import hrem_loss
        return hrem_loss(bb, out, ci, cfg, head_only) + out.get("aux_loss", 0.0)
    if out.get("hrem"):
        from .hrem import hrem_loss
        return hrem_loss(bb, out, ci, cfg, head_only)
    m = out["m"]
    L = F.cross_entropy(out["logits"], m["tgt_pi"])
    if "short" in out:
        pos = out["top"] == m["tgt_pi"].unsqueeze(1)
        hit = pos.any(1)                        # only queries whose truth is in the shortlist
        if hit.any():
            lab = pos.float().argmax(1)
            L = L + F.cross_entropy(out["short"][hit], lab[hit])
    L = L + 0.5 * F.cross_entropy(out["act_logits"], m["tgt_act"])
    if (not head_only and "tgt_code" in m and getattr(bb, "leaf", None) is not None
            and hasattr(bb.leaf, "score_pairs")):
        L = L + 0.5 * bb.leaf(out["z1"], m["tgt_code"])
    return L


# =========================================================================== #
def sthgcn_filter(ci, qs, train_ids):
    """Keep queries whose TARGET POI and user occur in the training split -- the test set of
    the Massive-STEPS STHGCN fork (preprocess_fn.remove_unseen_user_poi + only_keep_last)."""
    if not qs:
        return qs
    rows = np.concatenate([ci.trails[i]["rows"] for i in train_ids]) if len(train_ids) else np.zeros(0, int)
    pois = set(np.unique(ci.row_pi[rows]).tolist())
    users = {ci.trails[i]["user_id"] for i in train_ids}
    return [(ti, j) for ti, j in qs
            if int(ci.row_pi[ci.trails[ti]["rows"][j]]) in pois and ci.trails[ti]["user_id"] in users]


def _rank_of(S, tgt, ok):
    """1-based rank of tgt under scores S (-inf = excluded); 0 where not applicable."""
    ar = torch.arange(S.size(0), device=S.device)
    st = S[ar, tgt]
    r = (S > st.unsqueeze(1)).sum(1) + 1
    return torch.where(ok & torch.isfinite(st), r, torch.zeros_like(r)).cpu().numpy()


def _cond_ranks(out, cond):
    m = out["m"]; tgt = m["tgt_pi"]; vis = out["visited"]
    ar = torch.arange(tgt.size(0), device=tgt.device)
    new_t = ~vis[ar, tgt]
    cond["exp_cond"].append(_rank_of(out["exp_cond"].float(), tgt, new_t))
    cond["ret_cond"].append(_rank_of(out["ret_cond"].float(), tgt, ~new_t))
    if out.get("e_lat") is not None:
        cond["e_lat"].append(_rank_of(out["e_lat"].float().masked_fill(vis, float("-inf")), tgt, new_t))
    sim = out.get("ptr_sim"); tok = m["tok_pi"]
    if sim is not None:
        hit = (tok == tgt.unsqueeze(1)) & (tok >= 0)
        s_ = sim.float().masked_fill(tok < 0, float("-inf"))
        best_pos = s_.masked_fill(~hit, float("-inf")).max(1).values
        # rank by distinct tokens: count tokens of OTHER POIs scoring above the best target token
        r = ((s_ > best_pos.unsqueeze(1)) & ~hit).sum(1) + 1
        cond["ptr"].append(torch.where(hit.any(1), r, torch.zeros_like(r)).cpu().numpy())
    a_t = m["tgt_act"].clamp(max=out["gate"].size(1) - 1)
    cond["pi"].append(torch.sigmoid(out["gate"][ar, a_t].float()).cpu().numpy())


@torch.no_grad()
def evaluate(bb, head, ci, qs, cfg, bs=None):
    bb.eval(); head.eval()
    bs = bs or cfg.ft_batch
    ranks, ya, pa, rev = [], [], [], []
    cond = {k: [] for k in ("exp_cond", "ret_cond", "e_lat", "ptr", "pi")}
    for s in range(0, len(qs), bs):
        out = _forward(bb, head, ci, qs[s:s + bs], False, _amp(cfg, ci.device), cfg.rerank_m)
        ranks.append(_ranks(out))
        if "exp_cond" in out:
            _cond_ranks(out, cond)
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
    # channel-conditional diagnostics (HREM / world model): is each channel any good
    # GIVEN the branch?  explore_cond = rank of a NEW target among the user's unvisited
    # POIs; ret_cond = rank of a revisit target among visited POIs; lat_explore = the
    # latent-only (world-model) match; ptr = latent pointer over context tokens;
    # pi_* = mean explore probability on explore / revisit targets (gate calibration)
    if cond["exp_cond"]:
        for key, name in (("exp_cond", "explore_cond"), ("ret_cond", "revisit_cond"),
                          ("e_lat", "lat_explore"), ("ptr", "ptr_revisit")):
            rk = np.concatenate(cond[key]) if cond[key] else np.zeros(0)
            rk = rk[rk > 0]
            if rk.size:
                met[f"{name}_acc@1"] = float((rk <= 1).mean())
                met[f"{name}_acc@10"] = float((rk <= 10).mean())
        if os.environ.get("GEMMA_DEBUG_COND"):
            for key in ("exp_cond", "ret_cond", "e_lat", "ptr"):
                rk = np.concatenate(cond[key]) if cond[key] else np.zeros(0)
                print("DEBUG", key, "n=", int((rk > 0).sum()), "median rank=", np.median(rk[rk > 0]) if (rk > 0).any() else None)
        pis = np.concatenate(cond["pi"]); rv = rev.astype(bool)
        if (~rv).any(): met["pi_on_explore"] = float(pis[~rv].mean())
        if rv.any(): met["pi_on_revisit"] = float(pis[rv].mean())
    met["_ranks"] = ranks.astype(np.int32)
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
      gtrand100  : ground truth + 100 random POIs = the Massive-STEPS LLM-Move protocol
                   (src/prompts.py: random.sample(all POIs, 100) + target, sorted by distance)"""
    bb.eval(); head.eval()
    by_user = defaultdict(list)
    for ti in test_trails:
        if len(ci.trails[ti]["rows"]) >= 2:
            by_user[ci.trails[ti]["user_id"]].append(ti)
    users = sorted(by_user)
    variants = ("near100", "gtnear100", "gtrand100")
    res = {v: [] for v in variants}
    use_ar = (ci.codes is not None and hasattr(getattr(bb, "leaf", None), "score_pairs")
              and hasattr(head, "alpha_ar"))
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
    from .utils import amp_dtype_for
    return amp_dtype_for(device)


# =========================================================================== #
def finetune(bb, head, jobs, cfg, device, logger, epochs, lr_head, lr_bb,
             val=None, tag=""):
    """Train head (+ backbone when lr_bb>0) on one or more (CityIndex, queries).
    jobs: list of (ci, queries); batches of different cities are interleaved.
    val : (ci, queries) or a list of them -> model selection on acc@1.  The
          initialisation is a candidate, validation runs every cfg.val_every
          steps (and at each epoch end), early stop after cfg.patience checks
          without improvement."""
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
    resid = [p for n_, p in head.named_parameters()
             if (n_.startswith(("poi_res", "user_res")) or ".poi_res" in n_ or ".user_res" in n_)
             and p.requires_grad]
    rid = {id(p) for p in resid}
    core = [p for p in head.parameters() if p.requires_grad and id(p) not in rid]
    lr_res = getattr(cfg, "ft_lr_resid", lr_head)
    groups = [{"params": core, "lr": lr_head, "weight_decay": 1e-4, "base": lr_head}]
    if resid:   # per-POI / per-user residuals memorise fastest -> lower LR + decay
        groups.append({"params": resid, "lr": lr_res, "weight_decay": 0.05, "base": lr_res})
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_bb, "weight_decay": 0.01, "base": lr_bb})
    opt = torch.optim.AdamW(groups)
    amp_dtype = _amp(cfg, device)
    # fp16 (Turing GPUs) needs loss scaling or small backbone gradients underflow to 0
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    rng = np.random.RandomState(cfg.seed)
    steps_ep = sum(math.ceil(min(len(q), cfg.ft_max_queries) / cfg.ft_batch) for _, q in jobs)
    total = max(1, epochs * steps_ep); step = 0
    log_every = max(1, getattr(cfg, "log_every", 200))
    val_every = getattr(cfg, "val_every", 0)
    patience = getattr(cfg, "patience", 3)
    logger.info(f"[{tag}] device={device} amp={amp_dtype} steps/epoch={steps_ep} epochs={epochs} "
                f"lr_head={lr_head} lr_resid={lr_res if resid else '-'} lr_bb={lr_bb} "
                f"val_every={val_every or 'epoch'} patience={patience} "
                f"min_epochs={getattr(cfg, 'min_epochs', 0)}")
    vals = None if val is None else (val if isinstance(val, list) else [val])
    for ci_, _ in jobs:
        ci_.stats_off = False
    state = {"best": -1.0, "best_state": None, "bad": 0, "ep": 0}

    def snapshot():
        return (copy.deepcopy({k: v.cpu() for k, v in bb.state_dict().items()}),
                copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()}))

    def validate(where):
        vms = []
        for vv in vals:
            vci, vq = vv[0], vv[1]
            vci.stats_off = bool(vv[2]) if len(vv) > 2 else False
            vms.append(evaluate(bb, head, vci, vq, cfg))
            vci.stats_off = False
        vm = {k: float(np.mean([x[k] for x in vms])) for k in ("acc@1", "acc@5", "act@1")}
        improved = vm["acc@1"] > state["best"]
        if improved:
            state["best"], state["bad"], state["best_state"] = vm["acc@1"], 0, snapshot()
        else:
            state["bad"] += 1
        logger.info(f"[{tag}] val@{where}: acc@1={vm['acc@1']:.4f} acc@5={vm['acc@5']:.4f} "
                    f"act@1={vm['act@1']:.3f}{'  *best' if improved else ''}")
        bb.train() if tune_bb else bb.eval(); head.train()
        # no early stop during the first cfg.min_epochs epochs (slow-starting world-model
        # objectives would otherwise be cut off while the head has already plateaued)
        return state["bad"] >= patience and state["ep"] >= getattr(cfg, "min_epochs", 0)

    if vals is not None:                         # the initialisation is a candidate
        validate("init")
    stop = False
    for ep in range(epochs):
        state["ep"] = ep
        bb.train() if tune_bb else bb.eval(); head.train()
        order = []
        for k, (ci, q) in enumerate(jobs):
            idx = rng.permutation(len(q))[:cfg.ft_max_queries]
            order += [(k, idx[s:s + cfg.ft_batch]) for s in range(0, len(idx), cfg.ft_batch)]
        rng.shuffle(order)
        t0 = time.time(); tot = 0.0; n = 0; win = 0.0; wn = 0
        for it, (k, bidx) in enumerate(order):
            ci, q = jobs[k]
            for g in opt.param_groups:
                g["lr"] = g["base"] * 0.5 * (1 + math.cos(math.pi * step / total))
            sdrop = float(getattr(cfg, "stat_dropout", 0.0) or 0.0)
            ci.stats_off = bool(ci.has_stats and sdrop > 0 and rng.rand() < sdrop)
            out = _forward(bb, head, ci, [q[i] for i in bidx], True, amp_dtype, cfg.rerank_m)
            loss = _loss(bb, out, ci, cfg, head_only=not tune_bb)
            ci.stats_off = False
            if not torch.isfinite(loss):
                step += 1; continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); step += 1
            if hasattr(head, "after_step"):
                head.after_step(bb, cfg, tune_bb)
            lv = float(loss.detach()); tot += lv; n += 1; win += lv; wn += 1
            if (it + 1) % log_every == 0:
                el = time.time() - t0
                eta = el / (it + 1) * (len(order) - it - 1)
                logger.info(f"[{tag}] ep{ep} step {it + 1}/{len(order)} loss={win / wn:.4f} "
                            f"({el / (it + 1):.2f}s/step, epoch ETA {eta / 60:.1f} min) "
                            + (f"alpha_ar={head.alpha_ar.item():.2f}" if hasattr(head, "alpha_ar") else "")
                            + (" " + " ".join(f"{k}={v:.3f}" for k, v in head._last.items())
                               if getattr(head, "_last", None) else ""))
                win = 0.0; wn = 0
            if vals is not None and val_every and (it + 1) % val_every == 0:
                if validate(f"ep{ep}-step{it + 1}"):
                    stop = True; break
        logger.info(f"[{tag}] ft epoch {ep} loss={tot / max(1, n):.4f} ({time.time() - t0:.0f}s) | "
                    + describe_head(head, device))
        if stop:
            logger.info(f"[{tag}] early stop in epoch {ep}")
            break
        state["ep"] = ep + 1                     # epoch ep is complete
        if vals is not None and validate(f"ep{ep}-end"):
            logger.info(f"[{tag}] early stop at epoch {ep}")
            break
    if state["best_state"] is not None:
        bb.load_state_dict(state["best_state"][0]); head.load_state_dict(state["best_state"][1])
        logger.info(f"[{tag}] restored best checkpoint (val acc@1={state['best']:.4f})")
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.eval(); head.eval()
    return head


def describe_head(head, device):
    if hasattr(head, "describe"):
        return head.describe()
    with torch.no_grad():
        w = head.weights(torch.zeros(1, head.d_real, device=device))[0].tolist()
    return ("w0: " + " ".join(f"{k}={v:.2f}" for k, v in zip(FEATS, w))
            + f" alpha_ar={head.alpha_ar.item():.2f} tau={head.log_tau.exp().item():.1f}")


def new_head(bb, ci, cfg, device):
    args = (cfg.d_model, ci.name.size(1), bb.n_act, bb.n_pri, bb.n_sec,
            cfg.sid_levels if cfg.use_semantic_id else 0, cfg.sid_codebook, cfg.d_real)
    kind = getattr(cfg, "retriever", "hrem")
    if kind == "wm":
        from .hsjepa import WMHead
        return WMHead(*args, cfg).to(device)
    if kind == "hrem":
        from .hrem import HREMHead
        return HREMHead(*args).to(device)
    if kind == "hrem7":
        from .hrem import HREM7Head
        return HREM7Head(*args).to(device)
    return RealizationHead(*args).to(device)
