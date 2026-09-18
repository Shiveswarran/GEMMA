"""
gemma.semantic_id  (v2: collapse-hardened)
------------------------------------------
Residual-Quantized VAE for POI SEMANTIC IDs.  Diagnosis-driven fixes vs v1:
  * EMA codebook updates (VQ-VAE-2) instead of gradient updates -> stable, no collapse.
  * Positive diversity loss = KL(usage || uniform)  (v1 used -entropy -> negative-loss bug).
  * Tight bottleneck d (default 32) so quantization must be informative.
  * Per-epoch dead-code revival.
Feature normalization is done in build_semantic_ids.poi_features (block-balanced),
because the 1536-d LLM block (||.||=1) was being drowned by 7 geo/time/hier dims
(||.||=1.9) -> recon solved on 7 dims, codes collapsed.

  z=Enc(x); r0=z; for l: c_l=NN(r,e^l); r=r-e^l_{c_l}; z_q=sum e^l_{c_l}; x_hat=Dec(z_q)
  L = ||x-x_hat||^2 + beta*||z - sg[z_q]||^2 + gamma * sum_l KL(usage_l || Uniform)
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class RQVAE(nn.Module):
    def __init__(self, feat_dim, d=32, n_levels=3, codebook_size=256,
                 beta=0.25, gamma=1.0, ema=0.99):
        super().__init__()
        self.L, self.K, self.d = n_levels, codebook_size, d
        self.beta, self.gamma, self.ema = beta, gamma, ema
        self.enc = nn.Sequential(nn.Linear(feat_dim, 512), nn.GELU(),
                                 nn.Linear(512, 128), nn.GELU(), nn.Linear(128, d))
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.GELU(),
                                 nn.Linear(128, 512), nn.GELU(), nn.Linear(512, feat_dim))
        for l in range(n_levels):
            self.register_buffer(f"cb_{l}", torch.randn(codebook_size, d) * 0.02)
            self.register_buffer(f"ema_n_{l}", torch.zeros(codebook_size))
            self.register_buffer(f"ema_m_{l}", torch.zeros(codebook_size, d))
        self.register_buffer("_inited", torch.zeros(1))

    def cb(self, l): return getattr(self, f"cb_{l}")

    @torch.no_grad()
    def kmeans_init(self, x):
        z = self.enc(x); r = z
        for l in range(self.L):
            n = min(self.K, r.size(0))
            idx = torch.randperm(r.size(0), device=r.device)[:n]
            cb = r[idx].clone()
            if n < self.K:
                pad = cb[torch.randint(0, n, (self.K - n,), device=r.device)]
                cb = torch.cat([cb, pad + 0.01 * torch.randn_like(pad)], 0)
            getattr(self, f"cb_{l}").copy_(cb)
            getattr(self, f"ema_m_{l}").copy_(cb); getattr(self, f"ema_n_{l}").fill_(1.0)
            d2 = (r.pow(2).sum(1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(1)[None])
            r = r - cb[d2.argmin(1)]
        self._inited.fill_(1.0)

    def _assign(self, r, l):
        cb = self.cb(l)
        d2 = (r.pow(2).sum(1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(1)[None])
        return d2.argmin(1)

    @torch.no_grad()
    def _ema_update(self, r, c, l):
        oh = F.one_hot(c, self.K).type_as(r)
        emn = getattr(self, f"ema_n_{l}"); emm = getattr(self, f"ema_m_{l}")
        emn.mul_(self.ema).add_(oh.sum(0), alpha=1 - self.ema)
        emm.mul_(self.ema).add_(oh.t() @ r, alpha=1 - self.ema)
        getattr(self, f"cb_{l}").copy_(emm / emn.clamp_min(1e-5).unsqueeze(1))

    def forward(self, x, update=True):
        z = self.enc(x)
        r = z; embs = []; codes = []; kl = x.new_zeros(())
        logU = torch.log(torch.tensor(1.0 / self.K, device=x.device))
        for l in range(self.L):
            c = self._assign(r, l)
            if update and self.training:
                self._ema_update(r.detach(), c, l)
            e = self.cb(l)[c]; embs.append(e); codes.append(c); r = r - e
            p = torch.bincount(c, minlength=self.K).float()
            p = p / p.sum().clamp_min(1e-6)
            kl = kl + (p * (p.clamp_min(1e-9).log() - logU)).sum()
        z_q = torch.stack(embs, 0).sum(0)
        z_st = z + (z_q - z).detach()
        x_hat = self.dec(z_st)
        L_recon = F.mse_loss(x_hat, x)
        L_commit = F.mse_loss(z, z_q.detach())
        loss = L_recon + self.beta * L_commit + self.gamma * kl
        codes = torch.stack(codes, 1)
        return {"loss": loss, "recon": L_recon.detach(), "commit": L_commit.detach(),
                "kl": kl.detach(), "codes": codes,
                "usage": float((codes[:, 0].unique().numel()) / self.K)}

    @torch.no_grad()
    def revive_dead(self, x):
        z = self.enc(x); r = z
        for l in range(self.L):
            c = self._assign(r, l)
            dead = torch.where(torch.bincount(c, minlength=self.K) == 0)[0]
            if len(dead):
                pick = z[torch.randint(0, z.size(0), (len(dead),), device=z.device)]
                getattr(self, f"cb_{l}")[dead] = pick + 0.01 * torch.randn_like(pick)
                getattr(self, f"ema_m_{l}")[dead] = getattr(self, f"cb_{l}")[dead]
                getattr(self, f"ema_n_{l}")[dead] = 1.0
            r = r - self.cb(l)[self._assign(r, l)]

    @torch.no_grad()
    def encode_codes(self, x):
        r = self.enc(x); out = []
        for l in range(self.L):
            c = self._assign(r, l); out.append(c); r = r - self.cb(l)[c]
        return torch.stack(out, 1)
