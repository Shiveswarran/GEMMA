"""
gemma.backbone -- GEMMA v2 foundation model (self-supervised, JEPA).

One forward over a per-user diary:
  * context encoder f_theta (online) encodes the masked diary,
  * EMA target encoder f_xi (momentum, stop-grad) encodes the full diary,
  * predictor maps context summary + position offset -> predicted latent of each
    masked future check-in (JEPA latent prediction, loss = MSE),
  * anti-collapse regularizer on context states (VICReg / SIGReg),
  * hierarchical activity heads (activity -> primary -> secondary) decode the
    predicted latent, using the REAL Foursquare taxonomy levels,
  * motif head: soft-assign the diary's activity-transition day-graph to a
    learnable motif codebook (>=17) -> a low-complexity, city-invariant prior.

`extract(...)` returns frozen per-check-in features for downstream models
(GEMMA-as-backbone), the interface every downstream adapter consumes.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularizers import sigreg, vicreg


class CheckinEmbed(nn.Module):
    def __init__(self, cont_dim, n_act, n_pri, n_sec, d, dropout):
        super().__init__()
        self.proj = nn.Linear(cont_dim, d)
        self.act_emb = nn.Embedding(n_act, d)
        self.pri_emb = nn.Embedding(n_pri, d)
        self.sec_emb = nn.Embedding(n_sec, d)
        self.type_emb = nn.Embedding(3, d)                  # checkin/SEP/USR
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, cont, act, pri, sec, types):
        cont = torch.nan_to_num(cont)
        x = (self.proj(cont) + self.act_emb(act) + self.pri_emb(pri)
             + self.sec_emb(sec) + self.type_emb(types))
        return self.drop(self.norm(x))


class SeqEncoder(nn.Module):
    def __init__(self, d, n_layers, n_heads, ffn_mult, dropout, max_len=256):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, d * ffn_mult, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x, mask):
        L = x.size(1)
        x = x + self.pos[:L].unsqueeze(0)
        pad = mask < 0.5                                    # True = ignore
        h = self.enc(x, src_key_padding_mask=pad)
        summ = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1e-6)
        return h, summ


class Predictor(nn.Module):
    def __init__(self, d, n_layers, dropout, max_offset=256):
        super().__init__()
        self.off = nn.Embedding(max_offset + 1, d)
        layer = nn.TransformerEncoderLayer(d, 4, d * 2, dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, summ, offsets):
        B, T = offsets.shape
        q = summ.unsqueeze(1).expand(B, T, -1) + self.off(offsets.clamp(max=self.off.num_embeddings - 1))
        return self.enc(q)


class MotifCodebook(nn.Module):
    """Soft-assign a diary's activity-transition signature to M motif prototypes.
    Signature = flattened first-order activity transition matrix of the diary."""
    def __init__(self, n_act, n_motifs, temp=0.5):
        super().__init__()
        self.n_act = n_act
        self.proto = nn.Parameter(torch.randn(n_motifs, n_act * n_act) * 0.02)
        self.temp = temp

    def signature(self, act_seq, mask):
        # act_seq (B,L) ids, mask (B,L) checkin positions -> (B, n_act*n_act)
        B, L = act_seq.shape
        A = self.n_act
        sig = act_seq.new_zeros(B, A * A, dtype=torch.float32)
        for b in range(B):
            idx = torch.where(mask[b] > 0.5)[0]
            if len(idx) < 2:
                continue
            a = act_seq[b, idx]
            for i in range(len(a) - 1):
                u, v = int(a[i]), int(a[i + 1])
                if u != v:                                  # drop self-loops
                    sig[b, u * A + v] += 1.0
        n = sig.sum(1, keepdim=True).clamp_min(1e-6)
        return sig / n

    def forward(self, act_seq, mask):
        sig = self.signature(act_seq, mask).detach()
        P = F.normalize(self.proto, dim=-1)
        logits = F.normalize(sig, dim=-1) @ P.t() / self.temp
        q = F.softmax(logits, -1)
        # reconstruction: explain the day-graph by the motif mixture (sparsity)
        recon = q @ P
        rec_loss = F.mse_loss(recon, F.normalize(sig, dim=-1))
        usage = q.mean(0)
        ent = -(usage * torch.log(usage + 1e-8)).sum()      # maximize -> balanced
        return rec_loss - 0.01 * ent, q


class GEMMA(nn.Module):
    def __init__(self, cfg, cont_dim, n_act, n_pri, n_sec):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = CheckinEmbed(cont_dim, n_act, n_pri, n_sec, d, cfg.dropout)
        self.enc = SeqEncoder(d, cfg.n_ctx_layers, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
        self.embed_t = copy.deepcopy(self.embed)
        self.enc_t = copy.deepcopy(self.enc)
        for p in list(self.embed_t.parameters()) + list(self.enc_t.parameters()):
            p.requires_grad_(False)
        self.predictor = Predictor(d, cfg.n_pred_layers, cfg.dropout)
        self.tnorm = nn.LayerNorm(d, elementwise_affine=False)
        dh = d // 2
        self.dh = dh
        self.act_emb = nn.Embedding(n_act, dh)
        self.pri_emb = nn.Embedding(n_pri, dh)
        self.sec_emb = nn.Embedding(n_sec, dh)
        if cfg.geometry == "poincare":
            from .geometry import HyperbolicHead
            self.act_head = HyperbolicHead(d, n_act)
            self.pri_head = HyperbolicHead(d + dh, n_pri)
            self.sec_head = HyperbolicHead(d + dh, n_sec)
        else:
            self.act_head = nn.Linear(d, n_act)
            self.pri_head = nn.Linear(d + dh, n_pri)
            self.sec_head = nn.Linear(d + dh, n_sec)
        self.motif = MotifCodebook(n_act, cfg.n_motifs)
        # -- two-tower next-POI ranker --
        dt = cfg.d_tower
        self.name_dim = cfg.poi_emb_dim
        self.query_tower = nn.Sequential(nn.Linear(d, dt), nn.GELU(), nn.Linear(dt, dt))
        self.poi_tower = nn.Sequential(
            nn.Linear(self.name_dim + 3 * dh, dt), nn.GELU(), nn.Linear(dt, dt))
        self.n_act, self.n_pri, self.n_sec = n_act, n_pri, n_sec

    def _encode(self, embed, enc, b, mask):
        x = embed(b["cont"], b["act"], b["pri"], b["sec"], b["types"])
        return enc(x, mask)

    @torch.no_grad()
    def ema_update(self, m):
        for o, t in zip(self.embed.parameters(), self.embed_t.parameters()):
            t.mul_(m).add_(o, alpha=1 - m)
        for o, t in zip(self.enc.parameters(), self.enc_t.parameters()):
            t.mul_(m).add_(o, alpha=1 - m)

    def forward(self, b, use_reg=True):
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
        mse = ((pred - tgt) ** 2).mean(-1)
        L_pred = (mse * tgt_mask).sum() / tgt_mask.sum().clamp_min(1e-6)

        # hierarchy CE on masked targets (teacher-forced parents)
        m = tgt_mask.reshape(-1) > 0.5
        z = pred.reshape(-1, pred.size(-1))[m]
        ta = b["act"].reshape(-1)[m]; tp = b["pri"].reshape(-1)[m]; ts = b["sec"].reshape(-1)[m]
        if z.size(0) >= 2:
            la = self.act_head(z)
            lp = self.pri_head(torch.cat([z, self.act_emb(ta)], -1))
            ls = self.sec_head(torch.cat([z, self.pri_emb(tp)], -1))
            L_act = F.cross_entropy(la, ta); L_pri = F.cross_entropy(lp, tp); L_sec = F.cross_entropy(ls, ts)
            with torch.no_grad():
                act1 = (la.argmax(-1) == ta).float().mean()
                pri1 = (lp.argmax(-1) == tp).float().mean()
                sec1 = (ls.argmax(-1) == ts).float().mean()
        else:
            z0 = summ.new_zeros(())
            L_act = L_pri = L_sec = act1 = pri1 = sec1 = z0

        L_reg = (vicreg(ctx_states[ctx_mask > 0.5], cfg.vic_var, cfg.vic_cov)
                 if cfg.regularizer == "vicreg" else
                 sigreg(ctx_states[ctx_mask > 0.5], cfg.sig_n_proj)
                 if cfg.regularizer == "sigreg" else summ.new_zeros(()))
        L_mot, _ = self.motif(b["act"], full_mask)

        # -- two-tower InfoNCE (POI leaf): align predicted latent with true POI --
        if z.size(0) >= 2:
            name = b["cont"].reshape(-1, b["cont"].size(-1))[m][:, :self.name_dim]
            q = F.normalize(self.query_tower(z), dim=-1)
            p = self.poi_embed(name, ta, tp, ts)
            logits = (q @ p.t()) / self.cfg.nce_temp          # (M,M) in-batch
            # mask same-venue collisions (two anchors sharing the exact POI)
            vid = b["vid"].reshape(-1)[m] if "vid" in b else None
            labels = torch.arange(z.size(0), device=z.device)
            if vid is not None:
                same = (vid[:, None] == vid[None, :]) & (~torch.eye(z.size(0), dtype=torch.bool, device=z.device))
                logits = logits.masked_fill(same, float("-inf"))
            L_nce = F.cross_entropy(logits, labels)
            with torch.no_grad():
                poi1 = (logits.argmax(-1) == labels).float().mean()
        else:
            L_nce = summ.new_zeros(()); poi1 = summ.new_zeros(())

        total = (cfg.lambda_pred * L_pred + cfg.lambda_act * L_act
                 + cfg.lambda_pri * L_pri + cfg.lambda_sec * L_sec
                 + (cfg.lambda_reg * L_reg if use_reg else 0.0)
                 + cfg.lambda_motif * L_mot + cfg.lambda_nce * L_nce)
        return {"loss": total, "pred": L_pred.detach(), "act": L_act.detach(),
                "pri": L_pri.detach(), "sec": L_sec.detach(), "reg": L_reg.detach(),
                "motif": L_mot.detach(), "nce": L_nce.detach(),
                "act@1": act1.detach(), "pri@1": pri1.detach(),
                "sec@1": sec1.detach(), "poi@1": poi1.detach()}

    def poi_embed(self, name, act, pri, sec):
        x = torch.cat([name, self.act_emb(act), self.pri_emb(pri), self.sec_emb(sec)], -1)
        return F.normalize(self.poi_tower(x), dim=-1)

    def _hier_dists(self, z):
        """coarse->fine next-level distributions from a predicted latent."""
        act = F.softmax(self.act_head(z), -1); a = act.argmax(-1)
        pri = F.softmax(self.pri_head(torch.cat([z, self.act_emb(a)], -1)), -1); p = pri.argmax(-1)
        sec = F.softmax(self.sec_head(torch.cat([z, self.pri_emb(p)], -1)), -1)
        return act, pri, sec

    @torch.no_grad()
    def embed_prefix(self, b):
        """Inference: encode a prefix -> summary, next-level dists, query vector."""
        h, summ = self._encode(self.embed, self.enc, b, b["ctx_mask"])
        ones = torch.ones(summ.size(0), 1, dtype=torch.long, device=summ.device)
        z1 = self.tnorm(self.predictor(summ, ones))[:, 0]
        act, pri, sec = self._hier_dists(z1)
        q = F.normalize(self.query_tower(z1), dim=-1)
        return summ, act, pri, sec, q

    # -- downstream interface: frozen per-check-in contextual features ------
    @torch.no_grad()
    def extract(self, b):
        """Return (states h, summary) for a batch of prefixes (ctx_mask=all real).
        These are the features every downstream adapter consumes."""
        h, summ = self._encode(self.embed, self.enc, b, b["ctx_mask"])
        return h, summ
