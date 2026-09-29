"""
gemma.leaf_heads
----------------
Leaf-POI heads over the L-level semantic ID, selectable for ablation:

  retrieval  : two-tower InfoNCE (baseline; capped by feature information -- see docs)
  ar         : autoregressive over code tokens (GNPR-SID style)
  diffusion  : discrete absorbing-state masked-generative (MaskGIT/D3PM style)  [default]

The generative heads predict the semantic ID conditioned on the JEPA-predicted
latent z; at inference the predicted code selects a small candidate set (POIs
sharing the code prefix), then geo/temporal/revisit priors rank within it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK = -1                                                    # absorbing (masked) token


class CodeEmbed(nn.Module):
    def __init__(self, n_levels, K, d):
        super().__init__()
        self.L, self.K = n_levels, K
        # +1 row per level for the MASK token
        self.emb = nn.ModuleList([nn.Embedding(K + 1, d) for _ in range(n_levels)])
        self.lvl = nn.Parameter(torch.randn(n_levels, d) * 0.02)

    def forward(self, codes):                                # codes (B,L) with MASK=-1
        out = 0
        for l in range(self.L):
            idx = codes[:, l].clone()
            idx = torch.where(idx < 0, torch.full_like(idx, self.K), idx)  # MASK -> row K
            out = out + self.emb[l](idx) + self.lvl[l]
        return out                                           # (B,d) summed


class ARHead(nn.Module):
    def __init__(self, d, n_levels, K):
        super().__init__()
        self.L, self.K = n_levels, K
        self.emb = CodeEmbed(n_levels, K, d)
        self.step = nn.ModuleList([nn.Linear(d, K) for _ in range(n_levels)])
        self.tok = nn.ModuleList([nn.Embedding(K, d) for _ in range(n_levels)])

    def forward(self, z, codes):                             # teacher-forced CE
        loss = z.new_zeros(()); h = z
        for l in range(self.L):
            loss = loss + F.cross_entropy(self.step[l](h), codes[:, l])
            h = z + self.tok[l](codes[:, l])
        return loss / self.L

    @torch.no_grad()
    def generate(self, z):
        h = z; out = []
        for l in range(self.L):
            c = self.step[l](h).argmax(-1); out.append(c)
            h = z + self.tok[l](c)
        return torch.stack(out, 1)                           # (B,L)

    @torch.no_grad()
    def score_codes(self, z, codes, chunk=4096):
        """GNPR-SID-style ranking: log P(code | z) per candidate, teacher-forced
        with the candidate's OWN code.  z:(B,d), codes:(C,L) -> logprob:(B,C).
        Candidates are processed in chunks so the (B,C,K) tensor never blows up
        memory on large POI vocabularies."""
        B = z.size(0); C = codes.size(0)
        out = z.new_zeros(B, C)
        for s in range(0, C, chunk):
            cc = codes[s:s + chunk]                          # (c,L)
            c = cc.size(0)
            zc = z.unsqueeze(1).expand(B, c, -1)             # (B,c,d)
            h = zc; lp = z.new_zeros(B, c)
            for l in range(self.L):
                logp = F.log_softmax(self.step[l](h), dim=-1)   # (B,c,K)
                cl = cc[:, l]
                lp = lp + logp.gather(-1, cl.view(1, c, 1).expand(B, c, 1)).squeeze(-1)
                h = zc + self.tok[l](cl).unsqueeze(0)
            out[:, s:s + chunk] = lp
        return out


class DiffusionHead(nn.Module):
    """Discrete absorbing-state diffusion over code tokens (MaskGIT-style)."""
    def __init__(self, d, n_levels, K, steps=4):
        super().__init__()
        self.L, self.K, self.steps = n_levels, K, steps
        self.emb = CodeEmbed(n_levels, K, d)
        self.trunk = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.heads = nn.ModuleList([nn.Linear(d, K) for _ in range(n_levels)])

    def forward(self, z, codes):                             # random-mask denoising loss
        B = codes.size(0)
        r = torch.rand(B, 1, device=z.device)                # per-sample mask ratio
        keep = (torch.rand(B, self.L, device=z.device) > r)  # True=visible
        noisy = torch.where(keep, codes, torch.full_like(codes, MASK))
        h = self.trunk(z + self.emb(noisy))
        loss = z.new_zeros(()); n = 0
        for l in range(self.L):
            masked = ~keep[:, l]
            if masked.any():
                loss = loss + F.cross_entropy(self.heads[l](h)[masked], codes[masked, l])
                n += 1
        return loss / max(1, n)

    @torch.no_grad()
    def generate(self, z):
        B = z.size(0)
        codes = torch.full((B, self.L), MASK, device=z.device)
        for t in range(self.steps):
            h = self.trunk(z + self.emb(codes))
            for l in range(self.L):
                pred = self.heads[l](h).softmax(-1)
                conf, c = pred.max(-1)
                fill = codes[:, l] < 0
                codes[:, l] = torch.where(fill, c, codes[:, l])
        # ensure filled
        h = self.trunk(z + self.emb(codes))
        for l in range(self.L):
            fill = codes[:, l] < 0
            codes[:, l] = torch.where(fill, self.heads[l](h).argmax(-1), codes[:, l])
        return codes
