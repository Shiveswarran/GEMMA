"""Context encoder of GEMMA and its self-supervised pretraining objective.

The encoder reads a user's diary of identifier-free check-in tokens and
returns per-token states h_1..h_n, a pooled summary and, through a predictor, the latent z of
the *next* check-in.  It is pretrained on the source cities as a joint-embedding predictive
architecture (JEPA):

  * an online encoder sees the diary with some check-ins hidden,
  * an EMA target encoder sees the full diary (stop-gradient),
  * the predictor maps the online summary + target position to the target encoder's latent
    of every hidden check-in (MSE in latent space),
  * auxiliary heads decode the predicted latent into the activity -> primary -> leaf taxonomy,
    the place's semantic ID, the relative geometry and the time of day,
  * VICReg keeps the online states from collapsing; a motif codebook summarizes the diary's
    activity-transition graph (Schneider et al., 2013).

Only the online encoder, the predictor and the taxonomy / semantic-ID heads are used by the
next-place model (gemma/model.py).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- building blocks
class CheckinEmbed(nn.Module):
    """e_t = LN(W_c [text; time; geometry] + E_A[a] + E_P[p] + E_S[s] + E_type[type])."""

    def __init__(self, cont_dim, n_act, n_pri, n_sec, d, dropout):
        super().__init__()
        self.proj = nn.Linear(cont_dim, d)
        self.act_emb = nn.Embedding(n_act, d)
        self.pri_emb = nn.Embedding(n_pri, d)
        self.sec_emb = nn.Embedding(n_sec, d)
        self.type_emb = nn.Embedding(3, d)                  # check-in / [SEP] / reserved
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, cont, act, pri, sec, types):
        x = (self.proj(torch.nan_to_num(cont)) + self.act_emb(act) + self.pri_emb(pri)
             + self.sec_emb(sec) + self.type_emb(types))
        return self.drop(self.norm(x))


class SeqEncoder(nn.Module):
    def __init__(self, d, n_layers, n_heads, ffn_mult, dropout, max_len=256):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, d * ffn_mult, dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, x, mask):
        h = self.enc(x + self.pos[:x.size(1)].unsqueeze(0), src_key_padding_mask=mask < 0.5)
        summ = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1e-6)
        return h, summ


class Predictor(nn.Module):
    """Pooled context summary + embedding of the target position -> predicted latent."""

    def __init__(self, d, n_layers, dropout, max_offset=256):
        super().__init__()
        self.off = nn.Embedding(max_offset + 1, d)
        layer = nn.TransformerEncoderLayer(d, 4, d * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, summ, offsets):
        B, T = offsets.shape
        q = summ.unsqueeze(1).expand(B, T, -1) + self.off(offsets.clamp(max=self.off.num_embeddings - 1))
        return self.enc(q)


class ARCodeHead(nn.Module):
    """Autoregressive semantic-ID head: log p(code | z) = sum_l log p(c_l | z, c_<l)."""

    def __init__(self, d, n_levels, K):
        super().__init__()
        self.L, self.K = n_levels, K
        self.step = nn.ModuleList([nn.Linear(d, K) for _ in range(n_levels)])
        self.tok = nn.ModuleList([nn.Embedding(K, d) for _ in range(n_levels)])

    def forward(self, z, codes):                            # teacher-forced cross-entropy
        loss, h = z.new_zeros(()), z
        for l in range(self.L):
            loss = loss + F.cross_entropy(self.step[l](h), codes[:, l])
            h = z + self.tok[l](codes[:, l])
        return loss / self.L

    @torch.no_grad()
    def generate(self, z):
        h, out = z, []
        for l in range(self.L):
            c = self.step[l](h).argmax(-1); out.append(c)
            h = z + self.tok[l](c)
        return torch.stack(out, 1)


class MotifCodebook(nn.Module):
    """Soft assignment of a diary's activity-transition graph to M learnable motif prototypes.
    A HOME node is inserted at every trajectory boundary so that daily tours are closed."""

    def __init__(self, n_act, n_motifs, temp=0.5, dummy=True):
        super().__init__()
        self.dummy, self.HOME = dummy, n_act
        self.A = n_act + (1 if dummy else 0)
        self.proto = nn.Parameter(torch.randn(n_motifs, self.A * self.A) * 0.02)
        self.temp = temp

    def signature(self, act_seq, mask, types=None):
        B, L = act_seq.shape
        A = self.A
        sig = act_seq.new_zeros(B, A * A, dtype=torch.float32)
        for b in range(B):
            if self.dummy and types is not None:
                seq = [self.HOME]
                for j in range(L):
                    if mask[b, j] < 0.5 and types[b, j] != 1:
                        continue
                    seq.append(self.HOME if int(types[b, j]) == 1 else int(act_seq[b, j]))
                seq.append(self.HOME)
            else:
                seq = [int(act_seq[b, k]) for k in torch.where(mask[b] > 0.5)[0]]
            for u, v in zip(seq[:-1], seq[1:]):
                if u != v and u < A and v < A:
                    sig[b, u * A + v] += 1.0
        return sig / sig.sum(1, keepdim=True).clamp_min(1e-6)

    def forward(self, act_seq, mask, types=None):
        sig = F.normalize(self.signature(act_seq, mask, types).detach(), dim=-1)
        P = F.normalize(self.proto, dim=-1)
        q = F.softmax(sig @ P.t() / self.temp, -1)
        usage = q.mean(0)
        return F.mse_loss(q @ P, sig) + 0.01 * (usage * torch.log(usage + 1e-8)).sum(), q


class UncertaintyWeighter(nn.Module):
    """Kendall et al. task weighting with a floor; 'sid' keeps a fixed weight."""

    def __init__(self, names, fixed, min_w):
        super().__init__()
        self.fixed, self.min_w = dict(fixed), float(min_w)
        self.learned = [n for n in names if n not in self.fixed]
        self.log_sigma = nn.Parameter(torch.zeros(len(self.learned)))

    def forward(self, losses):
        total, parts = 0.0, {}
        for k, c in self.fixed.items():
            if k in losses:
                total = total + c * losses[k]; parts[f"w_{k}"] = float(c)
        for i, k in enumerate(self.learned):
            if k in losses:
                w = torch.clamp(torch.exp(-self.log_sigma[i]), min=self.min_w)
                total = total + w * losses[k] + self.log_sigma[i]
                parts[f"w_{k}"] = float(w.detach())
        return total, parts


def vicreg(z, var_w=1.0, cov_w=0.04, eps=1e-4):
    """Variance + covariance terms of VICReg on the online context states (M, d)."""
    if z.size(0) < 2:
        return z.new_zeros(())
    z = z - z.mean(0, keepdim=True)
    var_loss = F.relu(1.0 - torch.sqrt(z.var(0) + eps)).mean()
    cov = (z.T @ z) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return var_w * var_loss + cov_w * off.pow(2).sum() / z.size(1)


# ----------------------------------------------------------------------------- the encoder
class GEMMAEncoder(nn.Module):
    def __init__(self, cfg, cont_dim, n_act, n_pri, n_sec):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = CheckinEmbed(cont_dim, n_act, n_pri, n_sec, d, cfg.dropout)
        self.enc = SeqEncoder(d, cfg.n_ctx_layers, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
        self.embed_t = copy.deepcopy(self.embed)            # EMA target encoder
        self.enc_t = copy.deepcopy(self.enc)
        for p in list(self.embed_t.parameters()) + list(self.enc_t.parameters()):
            p.requires_grad_(False)
        self.predictor = Predictor(d, cfg.n_pred_layers, cfg.dropout)
        self.tnorm = nn.LayerNorm(d, elementwise_affine=False)
        dh = d // 2
        self.act_emb, self.pri_emb, self.sec_emb = (nn.Embedding(n, dh) for n in (n_act, n_pri, n_sec))
        self.act_head = nn.Linear(d, n_act)                 # p(a | z)
        self.pri_head = nn.Linear(d + dh, n_pri)            # p(p | z, a)
        self.sec_head = nn.Linear(d + dh, n_sec)            # p(s | z, p)
        self.motif = MotifCodebook(n_act, cfg.n_motifs, dummy=cfg.motif_dummy_nodes)
        self.use_sid = cfg.use_semantic_id
        self.leaf = ARCodeHead(d, cfg.sid_levels, cfg.sid_codebook) if self.use_sid else None
        self.name_dim = cfg.poi_emb_dim
        self.td = 6 * cfg.n_time_harmonics
        self.geo_head = nn.Linear(d, 3)                     # [log-distance, sin, cos bearing]
        self.temporal_head = nn.Linear(d, 2)                # [sin, cos hour]
        self.weighter = UncertaintyWeighter(
            ["jepa", "act", "pri", "sec", "sid", "geo", "tmp"], fixed={"sid": cfg.lambda_sid},
            min_w=cfg.uw_min_w) if cfg.uncertainty_weighting else None
        self.n_act, self.n_pri, self.n_sec = n_act, n_pri, n_sec

    # -------------------------------------------------------------- shared
    def _encode(self, embed, enc, b, mask):
        return enc(embed(b["cont"], b["act"], b["pri"], b["sec"], b["types"]), mask)

    def encode_next(self, b):
        """States h (B,L,d), summary (B,d) and the latent z (B,d) of the next position."""
        h, summ = self._encode(self.embed, self.enc, b, b["ctx_mask"])
        n = b["ctx_mask"].sum(1).long().clamp(min=1)
        z = self.tnorm(self.predictor(summ, n.unsqueeze(1)))[:, 0]
        return h, summ, z, n

    def path_logprob(self, z, cand_act, cand_pri, cand_sec):
        """log p(a_v | z) + log p(p_v | z, a_v) + log p(s_v | z, p_v) for every candidate v,
        with all parents enumerated (exact chain rule).  z: (B,d) -> (B,P)."""
        B, d = z.shape
        la = F.log_softmax(self.act_head(z), -1)
        A = la.size(1)
        za = z.unsqueeze(1).expand(B, A, d)
        lp = F.log_softmax(self.pri_head(torch.cat(
            [za, self.act_emb.weight.unsqueeze(0).expand(B, A, -1)], -1)), -1)        # (B,A,P)
        Pr = lp.size(2)
        zp = z.unsqueeze(1).expand(B, Pr, d)
        ls = F.log_softmax(self.sec_head(torch.cat(
            [zp, self.pri_emb.weight.unsqueeze(0).expand(B, Pr, -1)], -1)), -1)       # (B,P,S)
        return la[:, cand_act] + lp[:, cand_act, cand_pri] + ls[:, cand_pri, cand_sec]

    # -------------------------------------------------------------- pretraining
    @torch.no_grad()
    def ema_update(self, m):
        for o, t in zip(self.embed.parameters(), self.embed_t.parameters()):
            t.mul_(m).add_(o, alpha=1 - m)
        for o, t in zip(self.enc.parameters(), self.enc_t.parameters()):
            t.mul_(m).add_(o, alpha=1 - m)

    def forward(self, b, use_reg=True):
        """Self-supervised loss of one batch of diaries (see module docstring)."""
        cfg = self.cfg
        ctx_mask, tgt_mask = b["ctx_mask"], b["tgt_mask"]
        full_mask = ((ctx_mask + tgt_mask) > 0.5).float()
        ctx_states, summ = self._encode(self.embed, self.enc, b, ctx_mask)
        with torch.no_grad():
            tgt_states, _ = self._encode(self.embed_t, self.enc_t, b, full_mask)
        tgt = self.tnorm(tgt_states).detach()
        L = ctx_states.size(1)
        offsets = torch.arange(L, device=summ.device).unsqueeze(0).expand(summ.size(0), L)
        pred = self.tnorm(self.predictor(summ, offsets))
        L_pred = (((pred - tgt) ** 2).mean(-1) * tgt_mask).sum() / tgt_mask.sum().clamp_min(1e-6)

        m = tgt_mask.reshape(-1) > 0.5
        z = pred.reshape(-1, pred.size(-1))[m]
        ta, tp, ts = (b[k].reshape(-1)[m] for k in ("act", "pri", "sec"))
        cont = b["cont"].reshape(-1, b["cont"].size(-1))[m]
        losses, zero = {}, summ.new_zeros(())
        act1 = pri1 = sec1 = poi1 = zero
        if z.size(0) >= 2:
            la = self.act_head(z)
            lp = self.pri_head(torch.cat([z, self.act_emb(ta)], -1))
            ls = self.sec_head(torch.cat([z, self.pri_emb(tp)], -1))
            losses["act"] = F.cross_entropy(la, ta)
            losses["pri"] = F.cross_entropy(lp, tp)
            losses["sec"] = F.cross_entropy(ls, ts)
            with torch.no_grad():
                act1, pri1, sec1 = ((x.argmax(-1) == y).float().mean() for x, y in ((la, ta), (lp, tp), (ls, ts)))
            o = self.name_dim + self.td
            losses["geo"] = F.mse_loss(self.geo_head(z), cont[:, o:o + 3])
            losses["tmp"] = F.mse_loss(self.temporal_head(z), cont[:, self.name_dim:self.name_dim + 2])
            if self.leaf is not None and "code" in b:
                code = b["code"].reshape(-1, b["code"].size(-1))[m]
                losses["sid"] = self.leaf(z, code)
                with torch.no_grad():
                    poi1 = (self.leaf.generate(z) == code).all(-1).float().mean()
        losses["jepa"] = L_pred

        L_reg = (vicreg(ctx_states[ctx_mask > 0.5], cfg.vic_var, cfg.vic_cov)
                 if cfg.regularizer == "vicreg" else zero)
        L_mot, _ = self.motif(b["act"], full_mask, b.get("types"))
        if self.weighter is not None:
            core, wparts = self.weighter(losses)
        else:
            fixed = {"jepa": cfg.lambda_pred, "act": cfg.lambda_act, "pri": cfg.lambda_pri,
                     "sec": cfg.lambda_sec, "sid": cfg.lambda_sid, "geo": cfg.lambda_geo,
                     "tmp": cfg.lambda_temporal}
            core, wparts = sum(fixed[k] * v for k, v in losses.items()), {}
        total = core + (cfg.lambda_reg * L_reg if use_reg else 0.0) + cfg.lambda_motif * L_mot
        out = {"loss": total, "pred": L_pred.detach(), "reg": L_reg.detach(), "motif": L_mot.detach(),
               "act@1": act1, "pri@1": pri1, "sec@1": sec1, "poi@1": poi1}
        for k in ("act", "pri", "sec", "sid", "geo", "tmp"):
            out[k] = losses.get(k, zero).detach()
        out.update(wparts)
        return out


def load_encoder_state(model: nn.Module, state: dict):
    """Load a checkpoint, ignoring tensors of modules that the encoder does not use."""
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    if missing:
        raise KeyError(f"checkpoint is missing {len(missing)} encoder tensors, e.g. {missing[:5]}")
    model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)
