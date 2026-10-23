"""Anti-collapse regularizers for latent-predictive SSL (modern JEPA needs one).

vicreg : VICReg variance + covariance terms on the context embeddings (the JEPA
         latent-MSE plays the role of the invariance term), the current strong
         default for non-contrastive SSL.
sigreg : SIGReg / LeJEPA-style isotropic-Gaussian test via random projections
         (Epps-Pulley characteristic function).  Kept as an ablation.
"""
import torch
import torch.nn.functional as F


def vicreg(z, var_w=1.0, cov_w=0.04, eps=1e-4):
    """z: (M, d) real context embeddings."""
    if z.size(0) < 2:
        return z.new_zeros(())
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(1.0 - std).mean()                     # hinge at unit std
    d = z.size(1)
    cov = (z.T @ z) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / d
    return var_w * var_loss + cov_w * cov_loss


def sigreg(z, n_proj=128):
    if z.size(0) < 2:
        return z.new_zeros(())
    z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-5)
    W = torch.randn(z.size(1), n_proj, device=z.device)
    W = W / W.norm(dim=0, keepdim=True)
    p = z @ W                                                # (M, n_proj)
    t = torch.linspace(-3, 3, 16, device=z.device)
    emp = torch.exp(1j * (p[:, :, None] * t[None, None, :])).mean(0)  # (n_proj,16)
    ref = torch.exp(-0.5 * t ** 2)[None, :]
    return ((emp.real - ref) ** 2 + emp.imag ** 2).mean()
