"""
gemma.geometry -- Poincare ball (curvature c) for the hierarchy ablation.

Rationale: the activity->primary->secondary->POI hierarchy is tree-like, and
hyperbolic space embeds trees with low distortion.  With geometry='poincare',
the hierarchy heads become DISTANCE-to-prototype classifiers on the ball
(logit_k = -dist(z, proto_k)), trained jointly.  With 'euclidean' (default) the
heads are plain linear layers.  Everything is tested to run NaN-free; correct
Riemannian optimizer is optional (Euclidean AdamW on ball params works in
practice because expmap0 keeps points inside the ball).
"""
import torch
import torch.nn as nn

EPS = 1e-5


def _clamp_norm(x, maxn):
    n = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    cond = n > maxn
    return torch.where(cond, x / n * maxn, x)


def expmap0(v, c=1.0):
    """Map a tangent vector at origin into the Poincare ball."""
    sc = c ** 0.5
    vn = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    out = torch.tanh(sc * vn) * v / (sc * vn)
    return _clamp_norm(out, (1 - 1e-3) / sc)


def mobius_add(x, y, c=1.0):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / den.clamp_min(EPS)


def dist(x, y, c=1.0):
    sc = c ** 0.5
    diff = mobius_add(-x, y, c)
    dn = diff.norm(dim=-1).clamp(max=(1 - 1e-3) / sc)
    return (2 / sc) * torch.atanh((sc * dn).clamp(max=1 - EPS))


class HyperbolicHead(nn.Module):
    """Distance-to-prototype classifier on the Poincare ball."""
    def __init__(self, in_dim, n_classes, c=1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.proto = nn.Parameter(torch.randn(n_classes, in_dim) * 0.01)
        self.c = c

    def forward(self, z):
        z = expmap0(self.proj(z), self.c)                 # (B, d) on ball
        p = expmap0(self.proto, self.c)                   # (K, d) on ball
        # pairwise hyperbolic distance -> negative = logit
        d = dist(z.unsqueeze(1), p.unsqueeze(0), self.c)  # (B, K)
        return -d
