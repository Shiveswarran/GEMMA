"""Residual-quantized autoencoder (RQ-VAE) for place semantic IDs.

    z = Enc(x);  r_0 = z;  c_l = argmin_k ||r_l - e^l_k||;  r_{l+1} = r_l - e^l_{c_l}
    L = ||x - Dec(sum_l e^l_{c_l})||^2 + beta ||z - sg[z_q]||^2 + gamma sum_l KL(usage_l || U)

Codebooks are updated by exponential moving averages and dead codes are revived every epoch.
The model is trained on the places of the source cities only; places of every other city are
quantized with the frozen codebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RQVAE(nn.Module):
    def __init__(self, feat_dim, d=32, n_levels=3, codebook_size=256, beta=0.25, gamma=1.0, ema=0.99):
        super().__init__()
        self.L, self.K, self.d = n_levels, codebook_size, d
        self.beta, self.gamma, self.ema = beta, gamma, ema
        self.enc = nn.Sequential(nn.Linear(feat_dim, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(),
                                 nn.Linear(128, d))
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 512), nn.GELU(),
                                 nn.Linear(512, feat_dim))
        for l in range(n_levels):
            self.register_buffer(f"cb_{l}", torch.randn(codebook_size, d) * 0.02)
            self.register_buffer(f"ema_n_{l}", torch.zeros(codebook_size))
            self.register_buffer(f"ema_m_{l}", torch.zeros(codebook_size, d))

    def cb(self, l):
        return getattr(self, f"cb_{l}")

    def _assign(self, r, l):
        cb = self.cb(l)
        return (r.pow(2).sum(1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(1)[None]).argmin(1)

    @torch.no_grad()
    def kmeans_init(self, x):
        r = self.enc(x)
        for l in range(self.L):
            n = min(self.K, r.size(0))
            cb = r[torch.randperm(r.size(0), device=r.device)[:n]].clone()
            if n < self.K:
                pad = cb[torch.randint(0, n, (self.K - n,), device=r.device)]
                cb = torch.cat([cb, pad + 0.01 * torch.randn_like(pad)], 0)
            self.cb(l).copy_(cb)
            getattr(self, f"ema_m_{l}").copy_(cb); getattr(self, f"ema_n_{l}").fill_(1.0)
            r = r - cb[self._assign(r, l)]

    @torch.no_grad()
    def _ema_update(self, r, c, l):
        oh = F.one_hot(c, self.K).type_as(r)
        n_, m_ = getattr(self, f"ema_n_{l}"), getattr(self, f"ema_m_{l}")
        n_.mul_(self.ema).add_(oh.sum(0), alpha=1 - self.ema)
        m_.mul_(self.ema).add_(oh.t() @ r, alpha=1 - self.ema)
        self.cb(l).copy_(m_ / n_.clamp_min(1e-5).unsqueeze(1))

    def forward(self, x):
        z = self.enc(x)
        r, embs, codes, kl = z, [], [], x.new_zeros(())
        logU = torch.log(torch.tensor(1.0 / self.K, device=x.device))
        for l in range(self.L):
            c = self._assign(r, l)
            if self.training:
                self._ema_update(r.detach(), c, l)
            e = self.cb(l)[c]; embs.append(e); codes.append(c); r = r - e
            p = torch.bincount(c, minlength=self.K).float()
            p = p / p.sum().clamp_min(1e-6)
            kl = kl + (p * (p.clamp_min(1e-9).log() - logU)).sum()
        z_q = torch.stack(embs, 0).sum(0)
        x_hat = self.dec(z + (z_q - z).detach())
        L_rec, L_commit = F.mse_loss(x_hat, x), F.mse_loss(z, z_q.detach())
        codes = torch.stack(codes, 1)
        return {"loss": L_rec + self.beta * L_commit + self.gamma * kl, "recon": L_rec.detach(),
                "kl": kl.detach(), "codes": codes,
                "usage": float(codes[:, 0].unique().numel() / self.K)}

    @torch.no_grad()
    def revive_dead(self, x):
        z = self.enc(x); r = z
        for l in range(self.L):
            dead = torch.where(torch.bincount(self._assign(r, l), minlength=self.K) == 0)[0]
            if len(dead):
                pick = z[torch.randint(0, z.size(0), (len(dead),), device=z.device)]
                self.cb(l)[dead] = pick + 0.01 * torch.randn_like(pick)
                getattr(self, f"ema_m_{l}")[dead] = self.cb(l)[dead]
                getattr(self, f"ema_n_{l}")[dead] = 1.0
            r = r - self.cb(l)[self._assign(r, l)]

    @torch.no_grad()
    def encode_codes(self, x):
        r, out = self.enc(x), []
        for l in range(self.L):
            c = self._assign(r, l); out.append(c); r = r - self.cb(l)[c]
        return torch.stack(out, 1)
