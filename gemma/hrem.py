"""
gemma.hrem -- Hierarchical Return-Explore Mixture retriever (GEMMA-HREM).

Why (from the v5 diagnostics):
  * ~50% of next check-ins are RETURNS to a place in the user's own history and
    ~50% are EXPLORATIONS of a place new to that user.
  * The v5 realizer added all evidence into one score.  The history-visit
    evidence (weight ~2.2) then outranks every unvisited POI, so explore Acc@1
    was ~0 BY CONSTRUCTION, and next-activity fell below a frequency+Markov
    heuristic in all cities.

Model (exact, normalised, soft -- nothing is pruned):

  P(v | c) = P(a_v | c) * [ (1 - pi_a(c)) * P_ret(v | c, a_v)  +  pi_a(c) * P_exp(v | c, a_v) ]
             \\_activity_/   \\_____return to a known place____/    \\___explore a new place___/

  * P(a | c)       next ACTIVITY: backbone head on the world-model latent z, plus the
                   user's activity profile and activity transitions (so it cannot
                   lose to the frequency/Markov heuristics -- it contains them).
  * pi_a(c)        activity-conditioned explore gate.  Inputs: z, the user's
                   PER-ACTIVITY novelty rate (fraction of first visits), log S_u
                   (distinct places; the EPR law says P(explore) ~ rho * S^-gamma),
                   novelty of the current prefix.  Supervised by "is the target new?".
  * P_ret(v|c,a)   softmax over the user's VISITED POIs of activity a: recency-weighted
                   frequency, time-of-day and weekday affinity (Flashback-style),
                   recency of last visit, personal transitions from the last POI,
                   distance, and a LATENT POINTER: attention of the predicted latent z
                   onto the encoder states of the history tokens (world-model memory read).
  * P_exp(v|c,a)   softmax over UNVISITED POIs of activity a: hierarchical path
                   log p(primary|a,z) + log p(secondary|primary,z), latent match between
                   z and the POI's semantic-ID/LLM embedding, distance from the last
                   check-in, distance to the user's ANCHORS (top visited places),
                   and -- only when target data is allowed (few/full) -- global
                   transitions, popularity and time-of-day.

  The next-activity prediction IS P(a|c) (the hierarchy is the model, not a
  post-hoc filter), and the training loss -log P(v*|c) decomposes into
  activity + gate + within-channel terms.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .realize import FI, STAT_FEATS

RET = ["hist", "prefix", "is_last", "ptrans", "tod", "dow", "rec", "geo", "lptr"]
EXP = ["hier", "geo", "anchor", "covis", "pop", "hour"]
RET_INIT = {"hist": 1.0, "prefix": 0.5, "is_last": -1.0, "ptrans": 1.0, "tod": 0.5,
            "dow": 0.3, "rec": 1.0, "geo": 0.5, "lptr": 0.5}
EXP_INIT = {"hier": 0.5, "geo": 1.0, "anchor": 0.5, "covis": 1.0, "pop": 0.3, "hour": 0.3}
SIGNED = {"is_last", "lptr"}
# ---- GEMMA v7: population evidence in BOTH channels, opportunity-rank explore, momentum gate
RET7 = RET + ["ctr", "icf"]
EXP7 = ["hier", "geo", "anchor", "ctr", "ccat", "icf", "pop", "hour", "orank_sec", "orank_act"]
RET_INIT.update({"ctr": 0.5, "icf": 0.5})
EXP_INIT.update({"ctr": 0.5, "ccat": 0.3, "icf": 0.5, "orank_sec": 0.5, "orank_act": 0.3})
STAT7 = {"covis", "pop", "hour", "ctr", "ccat", "icf"}       # need the city log (few/full/zero_shot_u)


def _isp(y):
    return math.log(math.expm1(max(y, 1e-4)))


class HREMHead(nn.Module):
    kind = "hrem"

    RET, EXP = RET, EXP

    def __init__(self, d_model, name_dim, n_act, n_pri, n_sec, sid_levels, sid_K, d_real=256):
        super().__init__()
        self.d_real, self.n_act = d_real, n_act
        RET, EXP = self.RET, self.EXP
        # POI composition (semantic ID + LLM name + taxonomy) -- shared across cities
        self.name_proj = nn.Linear(name_dim, d_real)
        self.code_emb = nn.ModuleList([nn.Embedding(sid_K, d_real) for _ in range(sid_levels)])
        self.a_emb = nn.Embedding(n_act, d_real)
        self.p_emb = nn.Embedding(n_pri, d_real)
        self.s_emb = nn.Embedding(n_sec, d_real)
        self.ln = nn.LayerNorm(d_real)
        # query from the world model
        self.q_mlp = nn.Sequential(nn.Linear(3 * d_model, d_real), nn.GELU(), nn.Linear(d_real, d_real))
        self.q_exp = nn.Linear(d_real, d_real)
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))
        # latent pointer into the history memory
        self.ptr_q = nn.Linear(d_model, 64)
        self.ptr_k = nn.Linear(d_model, 64)
        # channel feature weights (context-gated, softplus = monotone evidence)
        self.wr0 = nn.Parameter(torch.tensor([RET_INIT[k] if k in SIGNED else _isp(RET_INIT[k]) for k in RET]))
        self.we0 = nn.Parameter(torch.tensor([EXP_INIT[k] if k in SIGNED else _isp(EXP_INIT[k]) for k in EXP]))
        self.gr = nn.Linear(d_real, len(RET)); self.ge = nn.Linear(d_real, len(EXP))
        for l in (self.gr, self.ge):
            nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        self.r_signed = torch.tensor([k in SIGNED for k in RET])
        self.e_signed = torch.tensor([k in SIGNED for k in EXP])
        # activity head: backbone logits + personal profile
        self.act_lin = nn.Linear(2 * n_act + 3, n_act)
        nn.init.zeros_(self.act_lin.weight); nn.init.zeros_(self.act_lin.bias)
        self.act_w = nn.Parameter(torch.tensor([0.5, 0.5]))          # scales for log-profile terms
        # activity-conditioned explore gate
        self.g_lin = nn.Linear(d_real, n_act)
        nn.init.zeros_(self.g_lin.weight); nn.init.zeros_(self.g_lin.bias)
        self.g_nov = nn.Parameter(torch.tensor(1.0))      # personal novelty-rate logit
        self.g_S = nn.Parameter(torch.tensor(-0.3))       # EPR: logit pi ~ log rho - gamma log S
        self.g_pref = nn.Parameter(torch.tensor(0.5))
        self.g_b = nn.Parameter(torch.zeros(n_act))
        self.poi_res = None; self.user_res = None
        self.use_emb = True

    # ---------------------------------------------------------------- #
    def attach_city(self, n_pois, n_users, device):
        self.poi_res = nn.Embedding(n_pois, self.d_real).to(device)
        self.user_res = nn.Embedding(n_users, self.d_real).to(device)
        nn.init.zeros_(self.poi_res.weight); nn.init.zeros_(self.user_res.weight)

    def poi_matrix(self, ci):
        e = self.name_proj(ci.name) + self.a_emb(ci.act.clamp(max=self.n_act - 1)) \
            + self.p_emb(ci.pri) + self.s_emb(ci.sec)
        if ci.codes is not None:
            for l, emb in enumerate(self.code_emb):
                if l < ci.codes.size(1):
                    e = e + emb(ci.codes[:, l].clamp(max=emb.num_embeddings - 1))
        e = self.ln(e)
        if self.poi_res is not None:
            e = e + self.poi_res.weight
        return e

    @staticmethod
    def _w(w0, gate, q, signed):
        w = w0 + gate(q)
        return torch.where(signed.to(q.device), w, F.softplus(w))

    def describe(self):
        with torch.no_grad():
            z = torch.zeros(1, self.d_real, device=self.wr0.device)
            wr = self._w(self.wr0, self.gr, z, self.r_signed)[0].tolist()
            we = self._w(self.we0, self.ge, z, self.e_signed)[0].tolist()
        extra = (f" mom={self.g_mom.item():.2f} far={self.g_far.item():.2f}" if hasattr(self, "g_mom") else "")
        return ("ret[" + " ".join(f"{k}={v:.2f}" for k, v in zip(self.RET, wr)) + "] exp[" +
                " ".join(f"{k}={v:.2f}" for k, v in zip(self.EXP, we)) +
                f"] gate[nov={self.g_nov.item():.2f} logS={self.g_S.item():.2f} "
                f"pref={self.g_pref.item():.2f}{extra}] tau={self.log_tau.exp().item():.1f}")


class HREM7Head(HREMHead):
    """GEMMA v7 head: v6 + population evidence (both channels), opportunity-rank explore
    features and a momentum/far-from-home explore gate.  Same mixture, more evidence."""
    kind = "hrem"
    RET, EXP = RET7, EXP7

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.g_mom = nn.Parameter(torch.tensor(0.0))     # last check-in was a first visit (bursty exploration)
        self.g_far = nn.Parameter(torch.tensor(0.0))     # log1p km from the user's top anchor


# ==================================================================== #
def _scatter(m, key, B, P, dv):
    out = torch.zeros(B, P, device=dv)
    b, p, v = m["sparse"][key]
    if b.numel():
        out.index_put_((b, p), v, accumulate=True)
    return out


def _group_lse(x, groups, A):
    """log-sum-exp of x (B,P) within each activity group -> (B,A)."""
    out = x.new_full((x.size(0), A), float("-inf"))
    for a, idx in enumerate(groups[:A]):
        if idx.numel():
            out[:, a] = torch.logsumexp(x[:, idx], dim=1)
    return out


def opportunity_logrank(ci, last_pi, groups):
    """-log rank of every POI among the POIs of its own group (category/activity), ordered by
    distance from the last check-in (Noulas et al. 2012: P(choose) ~ rank^-alpha, alpha ~ 0.8
    in every city -> a density-free, city-invariant geometry).  groups: (P,) long."""
    dv = ci.device; P = ci.P
    la = ci.lat[last_pi].unsqueeze(1); lo = ci.lon[last_pi].unsqueeze(1)
    dx = (ci.lon.unsqueeze(0) - lo) * torch.cos(torch.deg2rad(la)); dy = ci.lat.unsqueeze(0) - la
    d = dx * dx + dy * dy                                              # monotone in distance
    o1 = torch.argsort(d, dim=1)
    gs = groups[o1]
    o2 = torch.sort(gs, dim=1, stable=True).indices
    order = o1.gather(1, o2)
    counts = torch.bincount(groups, minlength=int(groups.max()) + 1)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(P, device=dv).unsqueeze(0).expand_as(order)
    r_sorted = (pos - starts[groups[order]] + 1).float()
    rank = torch.empty_like(r_sorted).scatter_(1, order, r_sorted)
    return -torch.log(rank)


def hrem_forward(bb, head, ci, qs, train, amp_dtype, with_target=True):
    bb_in, m = ci.batch(qs, with_target=with_target)
    dv = ci.device; B = m["B"]
    amp_on = dv.startswith("cuda") and amp_dtype is not None
    with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_on):
        h, summ = bb._encode(bb.embed, bb.enc, bb_in, bb_in["ctx_mask"])
        n = bb_in["ctx_mask"].sum(1).long().clamp(min=1)
        z1 = bb.tnorm(bb.predictor(summ, n.unsqueeze(1), h, bb_in["ctx_mask"]))[:, 0]
    h, summ, z1 = h.float(), summ.float(), z1.float()
    h_last = h[torch.arange(B, device=dv), n - 1]
    la_bb = F.log_softmax(bb.act_head(z1), -1)
    path = bb.path_logprob(z1, ci.act, ci.pri, ci.sec)
    hier_in = path - la_bb[:, ci.act]                                 # log p(pri|a)+log p(sec|pri)
    out = hrem_mixture(head, bb, ci, m, z_ptr=z1, h=h,
                       q_in=torch.cat([z1, h_last, summ], -1),
                       act_logits_wm=bb.act_head(z1), hier_in=hier_in)
    out["z1"] = z1
    out["key"] = torch.cat([z1, h_last, summ], -1)      # context key for city-memory retrieval
    return out


def hrem_mixture(head, bb, ci, m, z_ptr, h, q_in, act_logits_wm, hier_in, gate_extra=None):
    """Exact return-explore mixture given world-model quantities:
    z_ptr (B,d) latent used to read the history memory h (B,L,d); q_in (B,3d)
    query input; act_logits_wm (B,A) activity logits from the world model;
    hier_in (B,P) log p(primary|.)+log p(secondary|.) per POI; gate_extra (B,A)."""
    dv = ci.device; B, P = m["B"], ci.P
    A = bb.n_act
    # ---- shared evidence (realize.CityIndex.dense) + new behavioural features
    zero = torch.zeros(B, P, device=dv)
    base = ci.dense(m, zero)                                          # (B,P,9)
    hist_raw = _scatter(m, "hist", B, P, dv); pref_raw = _scatter(m, "prefix", B, P, dv)
    visited = (hist_raw > 0) | (pref_raw > 0)
    no_ev = getattr(ci.cfg, "evidence", "full") == "none"
    if no_ev:   # the return set is what the backbone can SEE: POIs of the context tokens
        tk_ = m["tok_pi"]
        visited = torch.zeros(B, P, dtype=torch.bool, device=dv)
        bi = torch.arange(B, device=dv).unsqueeze(1).expand_as(tk_)
        visited[bi[tk_ >= 0], tk_[tk_ >= 0]] = True
    tod = torch.log1p(_scatter(m, "tod", B, P, dv))
    dow = torch.log1p(_scatter(m, "dow", B, P, dv))
    rec = torch.zeros(B, P, device=dv)
    b_, p_, v_ = m["sparse"]["rec"]
    if b_.numel():
        rec[b_, p_] = v_
    # latent pointer: predicted latent attends to the encoder states of history tokens
    tok = m["tok_pi"]                                                 # (B,L) POI or -1
    sim = (head.ptr_q(z_ptr).unsqueeze(1) * head.ptr_k(h)).sum(-1) / 8.0  # (B,L)
    sim = torch.where(tok >= 0, sim, torch.full_like(sim, -1e4))
    lptr = torch.full((B, P + 1), -1e4, device=dv)
    lptr.scatter_reduce_(1, torch.where(tok >= 0, tok, torch.full_like(tok, P)), sim,
                         reduce="amax", include_self=True)
    lptr = lptr[:, :P]
    lptr = torch.where(lptr > -1e3, lptr, torch.zeros_like(lptr))
    # anchors: distance to the user's most visited places
    an = m["anchors"]                                                 # (B,5)
    alat, alon = ci.lat[an], ci.lon[an]                               # (B,5)
    dx = (ci.lon.view(1, P, 1) - alon.unsqueeze(1)) * 111.32 * torch.cos(torch.deg2rad(alat.unsqueeze(1)))
    dy = (ci.lat.view(1, P, 1) - alat.unsqueeze(1)) * 111.32
    anchor = -torch.log1p(torch.sqrt(dx * dx + dy * dy).min(-1).values)

    q = head.q_mlp(q_in)
    if head.user_res is not None:
        q = q + head.user_res(m["user"])

    RETk, EXPk = getattr(head, "RET", RET), getattr(head, "EXP", EXP)
    stat_on = ci.has_stats and not getattr(ci, "stats_off", False)
    zero_ = torch.zeros(B, P, device=dv)
    col = ci.collab_dense(m) if hasattr(ci, "collab_dense") and any(k in STAT7 for k in RETk + EXPk) else None
    col = col or {}
    feat = {"hist": base[..., FI["hist"]], "prefix": base[..., FI["prefix"]],
            "is_last": base[..., FI["is_last"]], "ptrans": base[..., FI["ptrans"]], "tod": tod,
            "dow": dow, "rec": rec, "geo": base[..., FI["geo"]], "lptr": lptr,
            "covis": base[..., FI["covis"]], "pop": base[..., FI["pop"]], "hour": base[..., FI["hour"]],
            "ctr": col.get("ctr", zero_), "ccat": col.get("ccat", zero_), "icf": col.get("icf", zero_)}
    if "orank_sec" in EXPk:
        feat["orank_sec"] = opportunity_logrank(ci, m["last_pi"], ci.sec)
    if "orank_act" in EXPk:
        feat["orank_act"] = opportunity_logrank(ci, m["last_pi"], ci.act)

    def _drop(w, names, key):
        drop = [k for k in str(getattr(ci.cfg, key, "") or "").replace(",", " ").split() if k]
        if drop:    # eval-time diagnostic: switch off chosen evidence channels
            w = w * torch.tensor([k not in drop for k in names], device=dv)
        return w

    # ---- RETURN channel (visited POIs only)
    fr = torch.stack([feat[k] for k in RETk], -1)
    wr = head._w(head.wr0, head.gr, q, head.r_signed)
    if no_ev:
        wr = wr * torch.tensor([k in ("lptr", "geo") for k in RETk], device=dv)
    if not stat_on:
        wr = wr * torch.tensor([k not in STAT7 for k in RETk], device=dv)
    wr = _drop(wr, RETk, "drop_ret")
    r = (fr * wr.unsqueeze(1)).sum(-1)
    NEG = torch.tensor(float("-inf"), device=dv)
    r = torch.where(visited, r, NEG)

    # ---- EXPLORE channel (unvisited POIs only)
    feat["hier"] = hier_in; feat["anchor"] = anchor
    fe = torch.stack([feat[k] for k in EXPk], -1)
    we = head._w(head.we0, head.ge, q, head.e_signed)
    if not stat_on:
        we = we * torch.tensor([k not in STAT7 for k in EXPk], device=dv)
    if no_ev:
        we = we * torch.tensor([k in ("hier", "geo") for k in EXPk], device=dv)
    we = _drop(we, EXPk, "drop_exp")
    e = (fe * we.unsqueeze(1)).sum(-1)
    e_lat = None
    if head.use_emb:
        E = head.poi_matrix(ci)
        e_lat = head.log_tau.exp() * (F.normalize(head.q_exp(q), dim=-1) @ F.normalize(E, dim=-1).t())
        e = e + e_lat
    e = torch.where(visited, NEG, e)

    # ---- ACTIVITY: backbone + personal profile (contains the heuristics)
    u_act, u_atr = m["u_act"][:, :A], m["u_atr"][:, :A]
    if no_ev:
        u_act, u_atr = torch.zeros_like(u_act), torch.zeros_like(u_atr)
    if u_act.size(1) < A:
        pad = A - u_act.size(1)
        u_act = F.pad(u_act, (0, pad)); u_atr = F.pad(u_atr, (0, pad))
    hour = m["hour"].float()
    tfe = torch.stack([torch.sin(2 * math.pi * hour / 24), torch.cos(2 * math.pi * hour / 24),
                       m["wkend"].float()], -1)
    pa = torch.log((u_act + 0.1) / (u_act.sum(1, keepdim=True) + 0.1 * A))
    pt = torch.log((u_atr + 0.1) / (u_atr.sum(1, keepdim=True) + 0.1 * A))
    if no_ev:
        pa, pt = torch.zeros_like(pa), torch.zeros_like(pt)
    act_logits = (act_logits_wm + head.act_w[0] * pa + head.act_w[1] * pt
                  + head.act_lin(torch.cat([torch.log1p(u_act), torch.log1p(u_atr), tfe], -1)))
    groups = ci.act_groups()
    present = torch.tensor([groups[a].numel() > 0 if a < len(groups) else False for a in range(A)],
                           device=dv)
    act_logits = act_logits.masked_fill(~present, float("-inf"))     # no mass on absent activities
    la = F.log_softmax(act_logits, -1)                                # (B,A)

    # ---- activity-conditioned explore gate (EPR prior + personal novelty)
    u_new = m["u_new"][:, :A]
    if u_new.size(1) < A:
        u_new = F.pad(u_new, (0, A - u_new.size(1)))
    nov = (u_new + 1.0) / (u_act + 2.0)                               # smoothed first-visit rate
    g = (head.g_lin(q) + head.g_nov * torch.logit(nov.clamp(1e-3, 1 - 1e-3))
         + head.g_S * torch.log1p(m["u_S"]).unsqueeze(1) + head.g_pref * m["u_pnew"].unsqueeze(1)
         + head.g_b)
    if hasattr(head, "g_mom") and not no_ev:
        a0 = m["anchors"][:, 0]
        ddx = (ci.lon[a0] - ci.lon[m["last_pi"]]) * 111.32 * torch.cos(torch.deg2rad(ci.lat[m["last_pi"]]))
        ddy = (ci.lat[a0] - ci.lat[m["last_pi"]]) * 111.32
        far = torch.log1p(torch.sqrt(ddx * ddx + ddy * ddy))
        g = g + head.g_mom * m["last_new"].unsqueeze(1) + head.g_far * far.unsqueeze(1)
    if no_ev:
        g = head.g_lin(q) + head.g_b
    if gate_extra is not None:
        g = g + gate_extra
    gm = str(getattr(ci.cfg, "gate_mode", "") or "")
    if gm:   # TEST-TIME gate ablations for the paper (weights are not retrained)
        if gm == "noS":            # drop the exploration-law term g_S * log S_u
            g = g - head.g_S * torch.log1p(m["u_S"]).unsqueeze(1)
        elif gm == "nopersonal":   # drop every personal term (novelty rate, log S_u, new prefix)
            g = head.g_lin(q) + head.g_b
        elif gm == "const":        # one global return/explore prior for everybody
            g = head.g_b.expand_as(g) if torch.is_tensor(head.g_b) else torch.full_like(g, float(head.g_b))
        else:
            raise ValueError(f"unknown gate_mode {gm!r}")
    lse_r = _group_lse(r, groups, A); lse_e = _group_lse(e, groups, A)
    has_r, has_e = torch.isfinite(lse_r), torch.isfinite(lse_e)
    lpi = torch.where(~has_r, torch.zeros_like(g), torch.where(~has_e, torch.full_like(g, float("-inf")), F.logsigmoid(g)))
    l1m = torch.where(~has_r, torch.full_like(g, float("-inf")), torch.where(~has_e, torch.zeros_like(g), F.logsigmoid(-g)))

    av = ci.act.clamp(max=A - 1)
    rr = torch.where(torch.isfinite(r), r - lse_r[:, av], NEG)
    ee = torch.where(torch.isfinite(e), e - lse_e[:, av], NEG)
    lp = la[:, av] + torch.logaddexp(l1m[:, av] + rr, lpi[:, av] + ee)

    out = {"logits": lp, "m": m, "act_logits": la, "gate": g, "visited": visited,
           "has_r": has_r, "has_e": has_e, "hrem": True,
           # channel-conditional scores (diagnostics + world-model retrieval losses)
           "ret_cond": la[:, av] + rr, "exp_cond": la[:, av] + ee,
           "e_lat": e_lat, "ptr_sim": sim,
           "wr": wr, "we": we}          # per-query evidence weights (analysis / interpretability)
    return out




def hrem_loss(bb, out, ci, cfg, head_only=False):
    m = out["m"]; tgt = m["tgt_pi"]; B = tgt.size(0)
    ar = torch.arange(B, device=tgt.device)
    lp_t = out["logits"][ar, tgt]
    ok = torch.isfinite(lp_t)
    L = -(lp_t[ok]).mean() if ok.any() else out["logits"].sum() * 0
    # auxiliary supervision of the gate: was the target a NEW place?
    a_t = ci.act[tgt].clamp(max=out["gate"].size(1) - 1)
    both = out["has_r"][ar, a_t] & out["has_e"][ar, a_t]
    if both.any():
        new_t = (~out["visited"][ar, tgt]).float()
        L = L + 0.3 * F.binary_cross_entropy_with_logits(out["gate"][ar, a_t][both], new_t[both])
    if (not head_only and "tgt_code" in m and getattr(bb, "leaf", None) is not None
            and hasattr(bb.leaf, "score_pairs")):
        L = L + 0.2 * bb.leaf(out["z1"], m["tgt_code"])     # keep the generative SID head alive
    return L
