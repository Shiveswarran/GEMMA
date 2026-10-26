"""GEMMA next-place model: the exact activity / return-explore mixture 

    P(v | xi) = P(a_v | xi) * [ (1 - pi_a(xi)) * P_R(v | xi, a_v) * 1[v known]
                                +     pi_a(xi)  * P_E(v | xi, a_v) * 1[v new]   ]

  P(a | xi)     next activity: encoder head on the next-step latent z + the user's activity
                profile and activity transitions (Eq. 6).
  pi_a(xi)      explore gate (Eq. 9): sigma(b_a + (G_pi q)_a + g_nu logit nu_{u,a}
                + g_S log(1 + S_u) + g_p u_new).  With only b and g_S it is the
                exploration-preferential-return law, logit pi = log rho - gamma log S.
  P_R           softmax over the user's known places of activity a, score w(q)^T phi(u, v).
  P_E           softmax over the remaining places of activity a,
                score w'(q)^T psi(u, v) + tau cos(W_e q, eps_v).
  w(q), w'(q)   softplus(w0 + G q): context-modulated, non-negative evidence weights
                (sign free for is_last and the latent pointer).

Every learned parameter acts on statistics, taxonomy classes, text embeddings or semantic
codes; none is indexed by a place, so the model applies unchanged to any city.  In the
fine-tuned setting, zero-initialized per-place and per-user residual embeddings are attached
(``attach_city``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RET = ["hist", "prefix", "is_last", "ptrans", "tod", "dow", "rec", "geo", "lptr"]
EXP = ["hier", "geo", "anchor", "covis", "pop", "hour"]
POPULATION = {"covis", "pop", "hour"}            # need population statistics (fine-tuned only)
SIGNED = {"is_last", "lptr"}
RET_INIT = {"hist": 1.0, "prefix": 0.5, "is_last": -1.0, "ptrans": 1.0, "tod": 0.5, "dow": 0.3,
            "rec": 1.0, "geo": 0.5, "lptr": 0.5}
EXP_INIT = {"hier": 0.5, "geo": 1.0, "anchor": 0.5, "covis": 1.0, "pop": 0.3, "hour": 0.3}


def _inv_softplus(y):
    return math.log(math.expm1(max(y, 1e-4)))


class MixtureHead(nn.Module):
    def __init__(self, d_model, text_dim, n_act, n_pri, n_sec, sid_levels, sid_K, d_real=256):
        super().__init__()
        self.d_real, self.n_act = d_real, n_act
        # content embedding eps_v of a place (text + taxonomy + semantic code), Eq. in Sec. 3.3
        self.name_proj = nn.Linear(text_dim, d_real)
        self.code_emb = nn.ModuleList([nn.Embedding(sid_K, d_real) for _ in range(sid_levels)])
        self.a_emb, self.p_emb, self.s_emb = (nn.Embedding(n, d_real) for n in (n_act, n_pri, n_sec))
        self.ln = nn.LayerNorm(d_real)
        # query q = MLP([z; h_n; mean h])
        self.q_mlp = nn.Sequential(nn.Linear(3 * d_model, d_real), nn.GELU(), nn.Linear(d_real, d_real))
        self.q_exp = nn.Linear(d_real, d_real)
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))
        # latent pointer: attention of z onto the encoder states of past visits
        self.ptr_q, self.ptr_k = nn.Linear(d_model, 64), nn.Linear(d_model, 64)
        # evidence weights w(q) = softplus(w0 + G q), G initialized at zero
        self.wr0 = nn.Parameter(torch.tensor([RET_INIT[k] if k in SIGNED else _inv_softplus(RET_INIT[k])
                                              for k in RET]))
        self.we0 = nn.Parameter(torch.tensor([EXP_INIT[k] if k in SIGNED else _inv_softplus(EXP_INIT[k])
                                              for k in EXP]))
        self.gr, self.ge = nn.Linear(d_real, len(RET)), nn.Linear(d_real, len(EXP))
        self.act_lin = nn.Linear(2 * n_act + 3, n_act)
        self.act_w = nn.Parameter(torch.tensor([0.5, 0.5]))
        # gate
        self.g_lin = nn.Linear(d_real, n_act)
        self.g_nov = nn.Parameter(torch.tensor(1.0))
        self.g_S = nn.Parameter(torch.tensor(-0.3))
        self.g_pref = nn.Parameter(torch.tensor(0.5))
        self.g_b = nn.Parameter(torch.zeros(n_act))
        for lin in (self.gr, self.ge, self.act_lin, self.g_lin):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.r_signed = torch.tensor([k in SIGNED for k in RET])
        self.e_signed = torch.tensor([k in SIGNED for k in EXP])
        self.poi_res = self.user_res = None
        self.use_emb = True

    def attach_city(self, n_pois, n_users, device):
        """Zero-initialized residual embeddings for the fine-tuned setting."""
        self.poi_res = nn.Embedding(n_pois, self.d_real).to(device)
        self.user_res = nn.Embedding(n_users, self.d_real).to(device)
        nn.init.zeros_(self.poi_res.weight); nn.init.zeros_(self.user_res.weight)

    def content(self, ci):
        """eps_v for every place of the city: (P, d_real)."""
        e = self.name_proj(ci.name) + self.a_emb(ci.act.clamp(max=self.n_act - 1)) \
            + self.p_emb(ci.pri) + self.s_emb(ci.sec)
        if ci.codes is not None:
            for l, emb in enumerate(self.code_emb):
                if l < ci.codes.size(1):
                    e = e + emb(ci.codes[:, l].clamp(max=emb.num_embeddings - 1))
        e = self.ln(e)
        return e + self.poi_res.weight if self.poi_res is not None else e

    @staticmethod
    def weights(w0, lin, q, signed):
        w = w0 + lin(q)
        return torch.where(signed.to(q.device), w, F.softplus(w))

    def describe(self):
        with torch.no_grad():
            z = torch.zeros(1, self.d_real, device=self.wr0.device)
            wr = self.weights(self.wr0, self.gr, z, self.r_signed)[0].tolist()
            we = self.weights(self.we0, self.ge, z, self.e_signed)[0].tolist()
        return ("return[" + " ".join(f"{k}={v:.2f}" for k, v in zip(RET, wr)) + "] explore[" +
                " ".join(f"{k}={v:.2f}" for k, v in zip(EXP, we)) +
                f"] gate[nu={self.g_nov.item():.2f} logS={self.g_S.item():.2f} "
                f"new={self.g_pref.item():.2f}] tau={self.log_tau.exp().item():.1f}")


def new_head(enc, ci, cfg, device):
    return MixtureHead(cfg.d_model, ci.name.size(1), enc.n_act, enc.n_pri, enc.n_sec,
                       cfg.sid_levels if cfg.use_semantic_id else 0, cfg.sid_codebook,
                       cfg.d_real).to(device)


# ----------------------------------------------------------------------------- forward
def _group_lse(x, groups, A):
    """log-sum-exp of x (B,P) within each activity -> (B,A)."""
    out = x.new_full((x.size(0), A), float("-inf"))
    for a, idx in enumerate(groups[:A]):
        if idx.numel():
            out[:, a] = torch.logsumexp(x[:, idx], dim=1)
    return out


def _switch_off(w, names, spec, device):
    drop = [k for k in str(spec or "").replace(",", " ").split() if k]
    return w * torch.tensor([k not in drop for k in names], device=device) if drop else w


def forward(enc, head, ci, qs, amp_dtype=None, with_target=True):
    """log P(v | xi) for every place of the city and a batch of queries."""
    enc_in, m = ci.batch(qs, with_target=with_target)
    dv, B, P, A = ci.device, m["B"], ci.P, enc.n_act
    cfg = ci.cfg
    amp_on = dv.startswith("cuda") and amp_dtype is not None
    with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_on):
        h, summ, z, n = enc.encode_next(enc_in)
    h, summ, z = h.float(), summ.float(), z.float()
    h_last = h[torch.arange(B, device=dv), n - 1]
    act_logits_enc = enc.act_head(z)
    hier = enc.path_logprob(z, ci.act, ci.pri, ci.sec) - F.log_softmax(act_logits_enc, -1)[:, ci.act]

    # ---- personal evidence
    f = ci.dense(m)
    tok = m["tok_pi"]
    no_ev = cfg.evidence == "none"
    if no_ev:     # ablation: the known places are only those the encoder sees
        visited = torch.zeros(B, P, dtype=torch.bool, device=dv)
        bi = torch.arange(B, device=dv).unsqueeze(1).expand_as(tok)
        visited[bi[tok >= 0], tok[tok >= 0]] = True
    else:
        visited = (ci.scatter(m, "hist") > 0) | (ci.scatter(m, "prefix") > 0)
    sim = (head.ptr_q(z).unsqueeze(1) * head.ptr_k(h)).sum(-1) / 8.0             # (B,L)
    sim = torch.where(tok >= 0, sim, torch.full_like(sim, -1e4))
    lptr = torch.full((B, P + 1), -1e4, device=dv)
    lptr.scatter_reduce_(1, torch.where(tok >= 0, tok, torch.full_like(tok, P)), sim,
                         reduce="amax", include_self=True)
    lptr = lptr[:, :P]
    f["lptr"] = torch.where(lptr > -1e3, lptr, torch.zeros_like(lptr))
    an = m["anchors"]
    alat, alon = ci.lat[an], ci.lon[an]
    dx = (ci.lon.view(1, P, 1) - alon.unsqueeze(1)) * 111.32 * torch.cos(torch.deg2rad(alat.unsqueeze(1)))
    dy = (ci.lat.view(1, P, 1) - alat.unsqueeze(1)) * 111.32
    f["anchor"] = -torch.log1p(torch.sqrt(dx * dx + dy * dy).min(-1).values)
    f["hier"] = hier

    q = head.q_mlp(torch.cat([z, h_last, summ], -1))
    if head.user_res is not None:
        q = q + head.user_res(m["user"])
    NEG = torch.tensor(float("-inf"), device=dv)

    # ---- return channel: known places only
    wr = head.weights(head.wr0, head.gr, q, head.r_signed)
    if no_ev:
        wr = wr * torch.tensor([k in ("lptr", "geo") for k in RET], device=dv)
    wr = _switch_off(wr, RET, cfg.drop_ret, dv)
    r = (torch.stack([f[k] for k in RET], -1) * wr.unsqueeze(1)).sum(-1)
    r = torch.where(visited, r, NEG)

    # ---- explore channel: unknown places only
    we = head.weights(head.we0, head.ge, q, head.e_signed)
    if not ci.has_stats:
        we = we * torch.tensor([k not in POPULATION for k in EXP], device=dv)
    if no_ev:
        we = we * torch.tensor([k in ("hier", "geo") for k in EXP], device=dv)
    we = _switch_off(we, EXP, cfg.drop_exp, dv)
    e = (torch.stack([f[k] for k in EXP], -1) * we.unsqueeze(1)).sum(-1)
    if head.use_emb:
        E = head.content(ci)
        e = e + head.log_tau.exp() * (F.normalize(head.q_exp(q), dim=-1) @ F.normalize(E, dim=-1).t())
    e = torch.where(visited, NEG, e)

    # ---- activity
    u_act, u_atr = m["u_act"][:, :A], m["u_atr"][:, :A]
    if no_ev:
        u_act, u_atr = torch.zeros_like(u_act), torch.zeros_like(u_atr)
    if u_act.size(1) < A:
        u_act, u_atr = F.pad(u_act, (0, A - u_act.size(1))), F.pad(u_atr, (0, A - u_atr.size(1)))
    hour = m["hour"].float()
    tfe = torch.stack([torch.sin(2 * math.pi * hour / 24), torch.cos(2 * math.pi * hour / 24),
                       m["wkend"].float()], -1)
    pa = torch.log((u_act + 0.1) / (u_act.sum(1, keepdim=True) + 0.1 * A))
    pt = torch.log((u_atr + 0.1) / (u_atr.sum(1, keepdim=True) + 0.1 * A))
    if no_ev:
        pa, pt = torch.zeros_like(pa), torch.zeros_like(pt)
    act_logits = (act_logits_enc + head.act_w[0] * pa + head.act_w[1] * pt
                  + head.act_lin(torch.cat([torch.log1p(u_act), torch.log1p(u_atr), tfe], -1)))
    groups = ci.act_groups()
    present = torch.tensor([a < len(groups) and groups[a].numel() > 0 for a in range(A)], device=dv)
    la = F.log_softmax(act_logits.masked_fill(~present, float("-inf")), -1)

    # ---- explore gate
    u_new = m["u_new"][:, :A]
    if u_new.size(1) < A:
        u_new = F.pad(u_new, (0, A - u_new.size(1)))
    nu = (u_new + 1.0) / (u_act + 2.0)                                # smoothed first-visit rate
    logS = torch.log1p(m["u_S"]).unsqueeze(1)
    g = (head.g_lin(q) + head.g_nov * torch.logit(nu.clamp(1e-3, 1 - 1e-3)) + head.g_S * logS
         + head.g_pref * m["u_pnew"].unsqueeze(1) + head.g_b)
    if no_ev:
        g = head.g_lin(q) + head.g_b
    if cfg.gate_mode == "noS":                 # test-time ablations (weights are not retrained)
        g = g - head.g_S * logS
    elif cfg.gate_mode == "nopersonal":
        g = head.g_lin(q) + head.g_b
    elif cfg.gate_mode == "const":
        g = head.g_b.expand_as(g)
    elif cfg.gate_mode:
        raise ValueError(f"unknown gate_mode {cfg.gate_mode!r}")

    # ---- exact mixture
    lse_r, lse_e = _group_lse(r, groups, A), _group_lse(e, groups, A)
    has_r, has_e = torch.isfinite(lse_r), torch.isfinite(lse_e)
    inf = torch.full_like(g, float("-inf")); zero = torch.zeros_like(g)
    log_pi = torch.where(~has_r, zero, torch.where(~has_e, inf, F.logsigmoid(g)))
    log_1m = torch.where(~has_r, inf, torch.where(~has_e, zero, F.logsigmoid(-g)))
    av = ci.act.clamp(max=A - 1)
    rr = torch.where(torch.isfinite(r), r - lse_r[:, av], NEG)
    ee = torch.where(torch.isfinite(e), e - lse_e[:, av], NEG)
    logp = la[:, av] + torch.logaddexp(log_1m[:, av] + rr, log_pi[:, av] + ee)
    return {"logits": logp, "m": m, "z": z, "act_logits": la, "gate": g, "visited": visited,
            "has_r": has_r, "has_e": has_e, "ret_cond": la[:, av] + rr, "exp_cond": la[:, av] + ee,
            "wr": wr, "we": we}


def loss_fn(enc, out, ci, train_encoder):
    """-log P(v* | xi) + 0.3 BCE(gate, target is new) [+ 0.2 semantic-ID loss of the encoder]."""
    m = out["m"]; tgt = m["tgt_pi"]; B = tgt.size(0)
    ar = torch.arange(B, device=tgt.device)
    lp = out["logits"][ar, tgt]
    ok = torch.isfinite(lp)
    L = -(lp[ok]).mean() if ok.any() else out["logits"].sum() * 0
    a_t = ci.act[tgt].clamp(max=out["gate"].size(1) - 1)
    both = out["has_r"][ar, a_t] & out["has_e"][ar, a_t]
    if both.any():
        new_t = (~out["visited"][ar, tgt]).float()
        L = L + 0.3 * F.binary_cross_entropy_with_logits(out["gate"][ar, a_t][both], new_t[both])
    if train_encoder and "tgt_code" in m and enc.leaf is not None:
        L = L + 0.2 * enc.leaf(out["z"], m["tgt_code"])
    return L
