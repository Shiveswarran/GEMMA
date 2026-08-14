"""
gemma.model
-----------
The GEMMA network.

Streams per check-in are concatenated into `cont` (name SBERT, category SBERT,
cyclic time, relative space) and combined with learned embeddings for the
(shared) category id, primary category id, token type, and a soft behavioural
role.  A Transformer *context encoder* summarises a prefix; an EMA *target
encoder* provides prediction targets; a *predictor* predicts masked-future
latents in representation space (JEPA).

Novelty (activity-based modelling made mechanical):
  * ActivityCodebook: K learnable activity prototypes.  Each check-in latent is
    softly assigned to prototypes -> an activity distribution q in the simplex.
    This is the learned, city-invariant "what activity is this" label.
  * Transition operator T (row-stochastic, K x K): an explicit, interpretable
    first-order activity grammar P(a_i | a_{i-1}) ("coffee -> office").
  * Grammar head: predicts the next-activity distribution from the context
    summary (higher-order grammar), trained by cross-entropy against the target
    encoder's assignment and softly tied to T q_prev (Markov consistency).

Losses returned by `forward`: pred (JEPA MSE), act (grammar CE), markov (KL),
sig (SIGReg isotropic-Gaussian), cb (codebook usage entropy).
"""
from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Token embedding (check-in -> d_model) with behavioural-role router         #
# --------------------------------------------------------------------------- #
class RoleRouter(nn.Module):
    """Soft behavioural-role assignment from a prefix of token embeddings.

    Attention-pools the sequence, scores it against R learnable role queries,
    and mixes R role embeddings.  Conditions activity transitions on a latent
    role (commuter / student / tourist ...) without observing it.
    """

    def __init__(self, d_model: int, n_roles: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_roles, d_model) * 0.02)
        self.role_emb = nn.Parameter(torch.randn(n_roles, d_model) * 0.02)
        self.pool = nn.Linear(d_model, d_model)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tok: (B,L,d), mask: (B,L) 1=real -> role vector (B,d) and dist (B,R)
        w = mask.unsqueeze(-1)
        pooled = (self.pool(tok) * w).sum(1) / w.sum(1).clamp_min(1e-6)  # (B,d)
        logits = pooled @ self.query.t()                                # (B,R)
        dist = F.softmax(logits, dim=-1)
        role = dist @ self.role_emb                                     # (B,d)
        return role, dist


class CheckinEmbed(nn.Module):
    def __init__(self, cont_dim: int, n_cat: int, n_primary: int,
                 d_model: int, n_roles: int, dropout: float):
        super().__init__()
        self.cont_proj = nn.Sequential(
            nn.Linear(cont_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cat_emb = nn.Embedding(n_cat, d_model)
        self.primary_emb = nn.Embedding(n_primary, d_model)
        self.type_emb = nn.Embedding(3, d_model)          # checkin / SEP / USR
        self.role = RoleRouter(d_model, n_roles)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cont, cid, pid, types, mask):
        cont = torch.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
        base = self.cont_proj(cont) + self.cat_emb(cid) \
            + self.primary_emb(pid) + self.type_emb(types)
        role, role_dist = self.role(base, mask)
        tok = self.norm(base + role.unsqueeze(1))
        return tok, role_dist


# --------------------------------------------------------------------------- #
#  Transformer encoder                                                        #
# --------------------------------------------------------------------------- #
class SeqEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ffn_mult, dropout, max_len=256):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ffn_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, tok, mask):
        L = tok.size(1)
        tok = tok + self.pos[:L].unsqueeze(0)
        # key_padding_mask: True = ignore
        kpm = mask < 0.5
        return self.enc(tok, src_key_padding_mask=kpm)


# --------------------------------------------------------------------------- #
#  Activity codebook + transition grammar                                     #
# --------------------------------------------------------------------------- #
class ActivityCodebook(nn.Module):
    """K learnable activity prototypes shared across cities.

    Both the JEPA-predicted latent (online) and the target-encoder latent
    (target) are assigned through the SAME prototypes, so that training to
    predict the target assignment shapes the prototypes to be *predictable* and
    *discriminative*.  Sinkhorn balancing on the target assignments prevents
    collapse onto a few prototypes.
    """

    def __init__(self, d_model: int, n_activities: int, temp: float):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(n_activities, d_model) * 0.02)
        self.temp = temp
        self.K = n_activities
        # explicit interpretable transition logits (K x K), row-stochastic
        self.trans = nn.Parameter(torch.zeros(n_activities, n_activities))

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1) @ F.normalize(self.proto, dim=-1).t()

    def assign(self, z: torch.Tensor, temp: float | None = None) -> torch.Tensor:
        t = self.temp if temp is None else temp
        return F.softmax(self.scores(z) / t, dim=-1)

    @staticmethod
    def sinkhorn(scores: torch.Tensor, eps: float = 0.05,
                 n_iter: int = 3) -> torch.Tensor:
        """SwAV-style optimal-transport assignment -> balanced soft targets."""
        with torch.no_grad():
            Q = torch.exp(scores / eps).t()          # K x N
            Q = Q / Q.sum().clamp_min(1e-8)
            K, N = Q.shape
            for _ in range(n_iter):
                Q = Q / Q.sum(1, keepdim=True).clamp_min(1e-8) / K
                Q = Q / Q.sum(0, keepdim=True).clamp_min(1e-8) / N
            Q = Q * N
            return Q.t()                             # N x K

    def markov_next(self, q_prev: torch.Tensor) -> torch.Tensor:
        T = F.softmax(self.trans, dim=-1)            # rows sum to 1
        return q_prev @ T


# --------------------------------------------------------------------------- #
#  Predictor + grammar head                                                   #
# --------------------------------------------------------------------------- #
class Predictor(nn.Module):
    """Predicts a masked target latent from (context summary, offset)."""

    def __init__(self, d_model, n_layers, dropout, max_offset=64):
        super().__init__()
        self.offset_emb = nn.Embedding(max_offset + 1, d_model)
        layers = []
        din = d_model * 2
        for _ in range(n_layers):
            layers += [nn.Linear(din, d_model), nn.GELU(), nn.Dropout(dropout)]
            din = d_model
        layers += [nn.Linear(d_model, d_model)]
        self.net = nn.Sequential(*layers)

    def forward(self, summary, offsets):
        # summary (B,d), offsets (B,T) -> (B,T,d)
        B, T = offsets.shape
        s = summary.unsqueeze(1).expand(B, T, -1)
        o = self.offset_emb(offsets.clamp(max=self.offset_emb.num_embeddings - 1))
        return self.net(torch.cat([s, o], dim=-1))


# --------------------------------------------------------------------------- #
#  SIGReg (isotropic-Gaussian regulariser, Epps-Pulley characteristic fn)     #
# --------------------------------------------------------------------------- #
def sigreg_loss(z: torch.Tensor, n_proj: int = 128,
                freqs=(0.5, 1.0, 1.5, 2.0)) -> torch.Tensor:
    """Penalise deviation of random 1-D projections of z from a standard
    Gaussian, using the empirical characteristic function (Epps-Pulley)."""
    if z.size(0) < 4:
        return z.new_zeros(())
    d = z.size(1)
    w = F.normalize(torch.randn(d, n_proj, device=z.device), dim=0)
    u = z @ w                                            # (M, n_proj)
    u = (u - u.mean(0, keepdim=True)) / (u.std(0, keepdim=True) + 1e-6)
    loss = z.new_zeros(())
    for t in freqs:
        target = math.exp(-0.5 * t * t)                 # CF of N(0,1) at t
        cos = torch.cos(t * u).mean(0)                  # real part
        sin = torch.sin(t * u).mean(0)                  # imag part
        loss = loss + ((cos - target) ** 2 + sin ** 2).mean()
    return loss / len(freqs)


# --------------------------------------------------------------------------- #
#  Full GEMMA module                                                          #
# --------------------------------------------------------------------------- #
class GEMMA(nn.Module):
    def __init__(self, cfg, cont_dim: int, n_cat: int, n_primary: int):
        super().__init__()
        self.cfg = cfg
        self.embed = CheckinEmbed(cont_dim, n_cat, n_primary, cfg.d_model,
                                  cfg.n_roles, cfg.dropout)
        self.context_enc = SeqEncoder(cfg.d_model, cfg.n_ctx_layers, cfg.n_heads,
                                      cfg.ffn_mult, cfg.dropout)
        # EMA target encoder = frozen copies of embed + encoder
        self.embed_t = copy.deepcopy(self.embed)
        self.context_enc_t = copy.deepcopy(self.context_enc)
        for p in list(self.embed_t.parameters()) + list(self.context_enc_t.parameters()):
            p.requires_grad_(False)

        self.predictor = Predictor(cfg.d_model, cfg.n_pred_layers, cfg.dropout,
                                   max_offset=cfg.max_user_seq_len)
        # normalise JEPA targets & predictions to a stable scale (I-JEPA trick):
        # keeps the latent-prediction loss O(1) and discourages the trivial
        # "shrink everything to zero" collapse.
        self.target_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.codebook = ActivityCodebook(cfg.d_model, cfg.n_activities,
                                         cfg.activity_temp)
        # explicit, transferable activity grammar: predict the next check-in's
        # (shared) primary category from the predicted latent.  This is the
        # headline "activity-top-1" diagnostic and the signal used at transfer.
        self.n_primary = n_primary
        self.cat_head = nn.Linear(cfg.d_model, n_primary)

    # -- context summary ----------------------------------------------------
    @staticmethod
    def _summary(states, mask):
        w = mask.unsqueeze(-1)
        return (states * w).sum(1) / w.sum(1).clamp_min(1e-6)

    def encode_context(self, cont, cid, pid, types, mask):
        tok, role_dist = self.embed(cont, cid, pid, types, mask)
        states = self.context_enc(tok, mask)
        return states, self._summary(states, mask), role_dist

    @torch.no_grad()
    def encode_target(self, cont, cid, pid, types, mask):
        tok, _ = self.embed_t(cont, cid, pid, types, mask)
        return self.context_enc_t(tok, mask)

    # -- EMA update ---------------------------------------------------------
    @torch.no_grad()
    def ema_update(self, decay: float):
        for pt, p in zip(self.embed_t.parameters(), self.embed.parameters()):
            pt.mul_(decay).add_(p, alpha=1 - decay)
        for pt, p in zip(self.context_enc_t.parameters(),
                         self.context_enc.parameters()):
            pt.mul_(decay).add_(p, alpha=1 - decay)

    # -- forward / loss -----------------------------------------------------
    def forward(self, batch: Dict, use_sig: bool = True) -> Dict:
        cfg = self.cfg
        # context
        ctx_states, summary, _ = self.encode_context(
            batch["ctx_cont"], batch["ctx_cid"], batch["ctx_pid"],
            batch["ctx_types"], batch["ctx_mask"])

        # target encoder sees full window [ctx ; tgt]; read out target slots
        full_cont = torch.cat([batch["ctx_cont"], batch["tgt_cont"]], dim=1)
        full_cid = torch.cat([batch["ctx_cid"], batch["tgt_cid"]], dim=1)
        full_pid = torch.cat([batch["ctx_pid"], batch["tgt_pid"]], dim=1)
        full_types = torch.cat([batch["ctx_types"],
                                torch.zeros_like(batch["tgt_cid"])], dim=1)
        full_mask = torch.cat([batch["ctx_mask"], batch["tgt_mask"]], dim=1)
        full_states = self.encode_target(full_cont, full_cid, full_pid,
                                         full_types, full_mask)
        Lc = batch["ctx_cont"].size(1)
        d = full_states.size(-1)
        tgt_latent = self.target_norm(full_states[:, Lc:])            # (B,T,d)
        tgt_mask = batch["tgt_mask"]                                  # (B,T)

        # -- JEPA latent prediction --------------------------------------
        pred = self.target_norm(self.predictor(summary, batch["offsets"]))
        mse = ((pred - tgt_latent.detach()) ** 2).mean(-1)           # (B,T)
        L_pred = (mse * tgt_mask).sum() / tgt_mask.sum().clamp_min(1e-6)

        # -- activity grammar: shared-prototype prediction (SwAV-style) ---
        m = tgt_mask.reshape(-1) > 0.5
        tgt_flat = tgt_latent.reshape(-1, d)[m].detach()             # (M,d)
        pred_flat = pred.reshape(-1, d)[m]                           # (M,d)
        if tgt_flat.size(0) >= 2:
            q_tgt = self.codebook.sinkhorn(
                self.codebook.scores(tgt_flat)).detach()            # (M,K)
            logp = F.log_softmax(
                self.codebook.scores(pred_flat) / cfg.activity_temp, dim=-1)
            L_act = -(q_tgt * logp).sum(-1).mean()
            with torch.no_grad():
                act_top1 = (logp.argmax(-1) == q_tgt.argmax(-1)).float().mean()
                q_mean = q_tgt.mean(0)
                L_cb = (q_mean * torch.log(q_mean + 1e-8)).sum()    # neg entropy
        else:
            L_act = summary.new_zeros(())
            act_top1 = summary.new_zeros(())
            L_cb = summary.new_zeros(())

        # -- transferable category grammar: predict next primary category -
        cat_logits = self.cat_head(pred)                            # (B,T,C)
        C = cat_logits.size(-1)
        cat_lp = F.log_softmax(cat_logits, dim=-1)
        ce_cat = F.nll_loss(cat_lp.reshape(-1, C),
                            batch["tgt_pid"].reshape(-1), reduction="none")
        L_cat = (ce_cat * tgt_mask.reshape(-1)).sum() / tgt_mask.sum().clamp_min(1e-6)
        with torch.no_grad():
            cat_hit = (cat_logits.argmax(-1) == batch["tgt_pid"]).float()
            cat_top1 = (cat_hit * tgt_mask).sum() / tgt_mask.sum().clamp_min(1e-6)

        # -- Markov consistency on the immediate next (offset==1) --------
        idx_last = (batch["ctx_mask"].sum(1) - 1).long().clamp_min(0)
        last_state = self.target_norm(
            ctx_states[torch.arange(ctx_states.size(0)), idx_last])
        q_prev = self.codebook.assign(last_state)                    # (B,K)
        markov = self.codebook.markov_next(q_prev)                   # (B,K)
        ones = torch.ones(summary.size(0), 1, dtype=torch.long, device=pred.device)
        pred1 = self.target_norm(self.predictor(summary, ones))[:, 0]
        phat1 = self.codebook.assign(pred1, temp=cfg.activity_temp)  # (B,K)
        L_markov = (phat1 * (torch.log(phat1 + 1e-8)
                             - torch.log(markov + 1e-8))).sum(-1).mean()

        # -- SIGReg on real context states -------------------------------
        if use_sig:
            real = ctx_states[batch["ctx_mask"] > 0.5]
            L_sig = sigreg_loss(real, cfg.sig_n_proj)
        else:
            L_sig = ctx_states.new_zeros(())

        total = (cfg.lambda_pred * L_pred + cfg.lambda_cat * L_cat
                 + cfg.lambda_act * L_act + cfg.lambda_markov * L_markov
                 + cfg.lambda_sig * L_sig + cfg.lambda_cb * L_cb)

        return {"loss": total, "pred": L_pred.detach(), "cat": L_cat.detach(),
                "act": L_act.detach(), "markov": L_markov.detach(),
                "sig": L_sig.detach(), "cb": L_cb.detach(),
                "cat_top1": cat_top1.detach(), "act_top1": act_top1.detach()}

    # -- inference helper: embed a prefix -> (summary, predicted next dist) --
    @torch.no_grad()
    def embed_prefix(self, batch):
        ctx_states, summary, _ = self.encode_context(
            batch["ctx_cont"], batch["ctx_cid"], batch["ctx_pid"],
            batch["ctx_types"], batch["ctx_mask"])
        ones = torch.ones(summary.size(0), 1, dtype=torch.long,
                          device=summary.device)
        pred1 = self.target_norm(self.predictor(summary, ones))[:, 0]
        cat_dist = F.softmax(self.cat_head(pred1), dim=-1)   # (B, n_primary)
        proto_dist = self.codebook.assign(pred1, temp=self.cfg.activity_temp)
        return summary, cat_dist, proto_dist
