"""
gemma.hsjepa -- HS-JEPA: Hierarchical Sequential JEPA world model for GEMMA.

One model, one objective family, trained on the SOURCE cities only; the target
city is never trained on in zero-shot.  Everything is self-supervised: every
target is derived from the check-in stream itself (no human labels).

Context  c  = [history trajectories] SEP ... SEP [current prefix]   (online encoder f_theta)

Hierarchical sequential predictor  g_phi  (cross-attends to the context states):
    for every future horizon h = 1..H   (next H check-ins)
      for every level  k = act -> pri -> sec -> place -> ret
          z_{h,k} = g_phi( [MASK] + pos(n+h) + level_k + horizon_h | c ,  z_{h',*} for h'<h ,  z_{h,<k} )
    i.e. coarse-to-fine inside a step and causal across steps (block-causal mask).

Hierarchical ABSTRACTION targets (EMA target encoder f_xi, stop-gradient):
    the true future check-in is shown to f_xi at four levels of detail
       act   : activity id + time            pri : + primary id
       sec   : + secondary id                place: full POI token (LLM name, geo, ids)
    y_{h,k} = LN( f_xi(c (+) x_{n+h}^{(k)}) )_{n+h}
    L_JEPA  = sum_k sum_h || z_{h,k} - sg(y_{h,k}) ||^2      (k over act, pri, sec, place)

    -> level k must encode exactly the information of its abstraction; the
       place latent is predicted CONDITIONED on the predicted activity/category
       latents (sequential hierarchy), the semantic analogue of LeCun's H-JEPA.

History infill (second masking view, train only): a random 15% of HISTORY
tokens are hidden from f_theta; g_phi predicts their place-level latents.

Behavioural read-outs on the predicted latents (all from the data itself):
    activity / primary / secondary probes on z_{h,act|pri|sec} (all horizons)
    return-vs-explore gate from z_{1,ret}

Retrieval IN the latent space (world-model retrieval, GEMMA-HREM mixture):
    P(v|c) = P(a_v|z_act) [ (1-pi_a(z_ret)) P_ret(v | memory read by z_place)
                            +   pi_a(z_ret)  P_exp(v | z_pri, z_sec, z_place, geo, anchors) ]
    The return channel is a pointer of the predicted place latent into the
    context-state memory; the explore channel scores unvisited POIs by the
    predicted category path and the match between z_place and the POI's
    semantic-ID / LLM embedding.

Total pretraining loss:
    L = -log P(v*|c)  + 0.3 BCE(gate)  + l_J L_JEPA + l_I L_infill + l_P L_probes + l_R VICReg(h)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrem import HREMHead, hrem_mixture
from .regularizers import sigreg, vicreg

LEVELS = ["act", "pri", "sec", "place", "ret"]
K_LAT = 4                                   # levels with latent targets (act..place)


class HPredictor(nn.Module):
    """Block-causal hierarchical predictor: queries (h, k) attend to the context
    states and to earlier queries (all levels of earlier horizons, coarser levels
    of the same horizon)."""

    def __init__(self, d, n_layers, H, n_heads=4, dropout=0.1, max_pos=320):
        super().__init__()
        self.H, self.K, self.max_pos = H, len(LEVELS), max_pos
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.pos = nn.Embedding(max_pos, d)
        self.lvl = nn.Embedding(self.K + 1, d)          # +1: infill query
        self.hor = nn.Embedding(H + 1, d)               # +1: infill query
        for e in (self.pos, self.lvl, self.hor):
            nn.init.normal_(e.weight, std=0.02)
        layer = nn.TransformerDecoderLayer(d, n_heads, 2 * d, dropout, batch_first=True,
                                           norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        T = H * self.K
        hh = torch.arange(T) // self.K
        kk = torch.arange(T) % self.K
        allowed = (hh[None, :] < hh[:, None]) | ((hh[None, :] == hh[:, None]) & (kk[None, :] <= kk[:, None]))
        self.register_buffer("blocked", ~allowed, persistent=False)
        self.register_buffer("hh", hh, persistent=False)
        self.register_buffer("kk", kk, persistent=False)

    def forward(self, summ, mem, mem_mask, n):
        B = summ.size(0)
        pos = (n.unsqueeze(1) + self.hh.unsqueeze(0)).clamp(max=self.max_pos - 1)
        q = (self.mask_tok + self.pos(pos) + self.lvl(self.kk).unsqueeze(0)
             + self.hor(self.hh).unsqueeze(0) + summ.unsqueeze(1))
        out = self.dec(q, mem, tgt_mask=self.blocked, memory_key_padding_mask=(mem_mask < 0.5))
        return out.view(B, self.H, self.K, -1)

    def infill(self, summ, mem, mem_mask, pos):
        q = (self.mask_tok + self.pos(pos.clamp(max=self.max_pos - 1))
             + self.lvl.weight[self.K] + self.hor.weight[self.H] + summ.unsqueeze(1))
        return self.dec(q, mem, memory_key_padding_mask=(mem_mask < 0.5))


class WMHead(nn.Module):
    """Everything on top of the GEMMA encoder: hierarchical predictor, probes,
    and the latent-space return-explore retriever."""
    kind = "wm"

    def __init__(self, d_model, name_dim, n_act, n_pri, n_sec, sid_levels, sid_K, d_real, cfg):
        super().__init__()
        self.H = int(getattr(cfg, "wm_H", 3))
        self.flat = bool(getattr(cfg, "wm_flat", False))
        self.infill_ratio = float(getattr(cfg, "wm_infill", 0.15))
        self.n_infill = 16
        self.lam = {"jepa": float(getattr(cfg, "lambda_wm_jepa", 1.0)),
                    "infill": float(getattr(cfg, "lambda_wm_infill", 0.5)),
                    "probe": float(getattr(cfg, "lambda_wm_probe", 0.5)),
                    "reg": float(getattr(cfg, "lambda_wm_reg", 0.1)),
                    # world-model retrieval losses: the predicted latent itself must
                    # (i) point at the right history token for returns and
                    # (ii) match the right NEW place for explorations -- gives the
                    # backbone a direct retrieval gradient even if the mixture
                    # learns to down-weight the latent channels (lptr->0 in v6)
                    "lat": float(getattr(cfg, "lambda_wm_lat", 0.5)),
                    "ptr": float(getattr(cfg, "lambda_wm_ptr", 0.3))}
        self.reg_kind = getattr(cfg, "regularizer", "vicreg")
        self.hpred = HPredictor(d_model, int(getattr(cfg, "wm_pred_layers", 3)), self.H,
                                dropout=cfg.dropout)
        self.act_probe = nn.Linear(d_model, n_act)
        self.pri_probe = nn.Linear(d_model, n_pri)
        self.sec_probe = nn.Linear(d_model, n_sec)
        self.gate_probe = nn.Linear(d_model, n_act)
        nn.init.zeros_(self.gate_probe.weight); nn.init.zeros_(self.gate_probe.bias)
        self.hrem = HREMHead(d_model, name_dim, n_act, n_pri, n_sec, sid_levels, sid_K, d_real)
        self.d_real = d_real
        self._last = {}

    # delegate realization utilities
    def attach_city(self, n_pois, n_users, device):
        self.hrem.attach_city(n_pois, n_users, device)

    def poi_matrix(self, ci):
        return self.hrem.poi_matrix(ci)

    @property
    def use_emb(self):
        return self.hrem.use_emb

    @use_emb.setter
    def use_emb(self, v):
        self.hrem.use_emb = v

    def after_step(self, bb, cfg, tune_bb):
        if tune_bb:
            bb.ema_update(cfg.ema_decay)                 # target encoder tracks the online one

    def describe(self):
        aux = " ".join(f"{k}={v:.3f}" for k, v in self._last.items())
        return f"wm[H={self.H} flat={self.flat} {aux}] " + self.hrem.describe()


# ------------------------------------------------------------------ #
def _future(ci, qs, H):
    rows = np.full((len(qs), H), -1, np.int64)
    for b, (ti, j) in enumerate(qs):
        r = ci.trails[ti]["rows"]
        for h in range(H):
            if j + h < len(r):
                rows[b, h] = r[j + h]
    return rows


def _full_seq(bb_in, ctx_mask0, n, cont_f, act_f, pri_f, sec_f, valid, H):
    """Context (all tokens visible) + the H future tokens appended at n..n+H-1."""
    B, L, cd = bb_in["cont"].shape
    dv = bb_in["cont"].device
    z = lambda t: torch.cat([t, t.new_zeros((B, H) + tuple(t.shape[2:]))], 1)
    cont, act, pri, sec, typ = z(bb_in["cont"]), z(bb_in["act"]), z(bb_in["pri"]), z(bb_in["sec"]), z(bb_in["types"])
    mask = z(ctx_mask0)
    pos = n.unsqueeze(1) + torch.arange(H, device=dv).unsqueeze(0)
    bi = torch.arange(B, device=dv).unsqueeze(1).expand(B, H)
    vf = valid.float()
    cont[bi, pos] = cont_f * vf.unsqueeze(-1)
    act[bi, pos] = act_f * valid; pri[bi, pos] = pri_f * valid; sec[bi, pos] = sec_f * valid
    typ[bi, pos] = 0
    mask[bi, pos] = vf
    return {"cont": cont, "act": act, "pri": pri, "sec": sec, "types": typ, "ctx_mask": mask}, pos


def wm_forward(bb, head, ci, qs, train, amp_dtype, with_target=True):
    bb_in, m = ci.batch(qs, with_target=with_target)
    dv = ci.device; B = m["B"]; H = head.H
    ar = torch.arange(B, device=dv)
    ctx0 = bb_in["ctx_mask"]
    n = ctx0.sum(1).long().clamp(min=1)                 # contexts are left-aligned
    amp_on = dv.startswith("cuda") and amp_dtype is not None
    do_aux = train and with_target and torch.is_grad_enabled()

    # ---- second masking view: hide random HISTORY tokens from the online encoder
    online = ctx0; inf_pos = inf_valid = None
    if do_aux and head.infill_ratio > 0:
        L = ctx0.size(1)
        posL = torch.arange(L, device=dv).unsqueeze(0)
        j = torch.as_tensor([q[1] for q in qs], device=dv)
        cand = (m["tok_pi"] >= 0) & (posL < (n - j).unsqueeze(1))
        r = torch.rand(B, L, device=dv)
        sel = cand & (r < head.infill_ratio)
        k = min(head.n_infill, L)
        vals, idx = torch.where(sel, r, torch.full_like(r, -1.0)).topk(k, dim=1)
        inf_pos, inf_valid = idx, vals >= 0
        online = ctx0.clone()
        online[ar.unsqueeze(1).expand(B, k)[inf_valid], idx[inf_valid]] = 0.0
    b_on = dict(bb_in); b_on["ctx_mask"] = online

    with torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_on):
        h, summ = bb._encode(bb.embed, bb.enc, b_on, online)
        zf = head.hpred(summ, h, online, n)                       # (B,H,K,d)
        zinf = head.hpred.infill(summ, h, online, inf_pos) if inf_pos is not None else None
    h, summ, zf = h.float(), summ.float(), bb.tnorm(zf.float())
    z_act, z_pri, z_sec, z_place, z_ret = (zf[:, 0, k] for k in range(5))

    lp_pri = F.log_softmax(head.pri_probe(z_pri), -1)
    lp_sec = F.log_softmax(head.sec_probe(z_sec), -1)
    hier_in = lp_pri[:, ci.pri.clamp(max=lp_pri.size(1) - 1)] + lp_sec[:, ci.sec.clamp(max=lp_sec.size(1) - 1)]
    out = hrem_mixture(head.hrem, bb, ci, m, z_ptr=z_place, h=h,
                       q_in=torch.cat([z_place, z_act, summ], -1),
                       act_logits_wm=head.act_probe(z_act), hier_in=hier_in,
                       gate_extra=head.gate_probe(z_ret))
    out["z1"] = z_place
    out["wm"] = True

    if do_aux:
        city = ci.city
        Fr = _future(ci, qs, H); valid_np = Fr >= 0; Fc = np.clip(Fr, 0, None)
        tt = lambda x, dt=None: torch.as_tensor(x, device=dv) if dt is None else torch.as_tensor(x, device=dv, dtype=dt)
        valid = tt(valid_np)
        cont_f = tt(city.cont[Fc].astype(np.float32))
        act_f = tt(np.asarray(city.act_id)[Fc]); pri_f = tt(np.asarray(city.pri_id)[Fc])
        sec_f = tt(np.asarray(city.sec_id)[Fc])
        nd, td = bb.name_dim, bb.td
        time_only = torch.zeros_like(cont_f); time_only[..., nd:nd + td] = cont_f[..., nd:nd + td]
        zero = torch.zeros_like(act_f)
        views = {"act": (time_only, act_f, zero, zero), "pri": (time_only, act_f, pri_f, zero),
                 "sec": (time_only, act_f, pri_f, sec_f), "place": (cont_f, act_f, pri_f, sec_f)}
        levels = ["place"] if head.flat else ["act", "pri", "sec", "place"]
        vm = valid.float()
        L_j = zf.new_zeros(()); y_inf = None
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_on):
            ys = {}
            for lv in levels:
                full, pos = _full_seq(bb_in, ctx0, n, *views[lv], valid, H)
                st, _ = bb._encode(bb.embed_t, bb.enc_t, full, full["ctx_mask"])
                st = bb.tnorm(st.float())
                ys[lv] = st[ar.unsqueeze(1).expand(B, H), pos]
                if lv == "place" and inf_pos is not None:
                    y_inf = st[ar.unsqueeze(1).expand_as(inf_pos), inf_pos]
        for lv in levels:
            k = LEVELS.index(lv)
            err = ((zf[:, :, k] - ys[lv]) ** 2).mean(-1)            # (B,H)
            L_j = L_j + (err * vm).sum() / vm.sum().clamp_min(1.0)
        L_j = L_j / len(levels)
        L_i = zf.new_zeros(())
        if zinf is not None and y_inf is not None and inf_valid.any():
            zi = bb.tnorm(zinf.float())
            ei = ((zi - y_inf) ** 2).mean(-1)
            L_i = (ei * inf_valid.float()).sum() / inf_valid.float().sum().clamp_min(1.0)
        # probes on every horizon (activity / primary / secondary of the next H check-ins)
        L_p = zf.new_zeros(())
        if valid.any():
            for probe, k, yv in ((head.act_probe, 0, act_f), (head.pri_probe, 1, pri_f),
                                 (head.sec_probe, 2, sec_f)):
                logits = probe(zf[:, :, k])[valid]
                L_p = L_p + F.cross_entropy(logits, yv[valid].clamp(max=logits.size(-1) - 1))
        hv = h[online > 0.5]
        L_r = (sigreg(hv) if head.reg_kind == "sigreg" else vicreg(hv)) if hv.size(0) > 1 else zf.new_zeros(())
        L_lat, L_ptr = _wm_retrieval_losses(out, m, ar)
        lam = head.lam
        out["aux_loss"] = (lam["jepa"] * L_j + lam["infill"] * L_i + lam["probe"] * L_p
                           + lam["reg"] * L_r + lam["lat"] * L_lat + lam["ptr"] * L_ptr)
        head._last = {"jepa": float(L_j.detach()), "infill": float(L_i.detach()),
                      "probe": float(L_p.detach()), "reg": float(L_r.detach()),
                      "lat": float(L_lat.detach()), "ptr": float(L_ptr.detach())}
    return out


def _wm_retrieval_losses(out, m, ar):
    """L_lat: CE of the LATENT-ONLY match over all city POIs (all targets) averaged with the
    same CE over the user's unvisited POIs (explore targets).
    L_ptr: revisit targets present in the context -> CE of the latent pointer over the
    context tokens (multi-positive: any occurrence of the target POI counts)."""
    tgt = m["tgt_pi"]; vis = out["visited"]
    zero = out["logits"].new_zeros(())
    L_lat = zero
    e_lat = out.get("e_lat")
    new_t = ~vis[ar, tgt]
    if e_lat is not None:
        # (a) contrastive / generative: predicted next-place latent vs the TEXT-grounded POI
        #     embeddings of ALL city POIs (InfoNCE over the city = mobility-CLIP)
        L_lat = F.cross_entropy(e_lat.float(), tgt)
        # (b) the same restricted to new places among the user's unvisited POIs (exploration)
        if new_t.any():
            s = e_lat[new_t].float().masked_fill(vis[new_t], float("-inf"))
            L_lat = 0.5 * (L_lat + F.cross_entropy(s, tgt[new_t]))
    L_ptr = zero
    sim = out.get("ptr_sim")
    if sim is not None:
        tok = m["tok_pi"]
        valid = tok >= 0
        pos = valid & (tok == tgt.unsqueeze(1))
        rows = pos.any(1)
        if rows.any():
            s = sim[rows].float()
            lse_all = torch.logsumexp(s.masked_fill(~valid[rows], float("-inf")), 1)
            lse_pos = torch.logsumexp(s.masked_fill(~pos[rows], float("-inf")), 1)
            L_ptr = (lse_all - lse_pos).mean()
    return L_lat, L_ptr
