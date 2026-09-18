"""
gemma.multitask -- Kendall homoscedastic-uncertainty weighting.

For per-task losses L_k with learnable log-variances s_k = log sigma_k^2:
    L = sum_k  exp(-s_k) * L_k  +  s_k
The exp(-s_k) auto-scales each task; the +s_k regularizer prevents s_k -> +inf.
Regression tasks (geo/temporal/jepa) conventionally use 0.5*exp(-s)*L + 0.5*s;
we use the unified exp(-s)*L + s form for all (differs only by a constant factor,
absorbed by the learned s), which is numerically stable and simpler to reason about.
"""
import torch
import torch.nn as nn


class UncertaintyWeighter(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.names = list(task_names)
        self.log_sigma = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, losses: dict):
        total = 0.0; parts = {}
        for i, k in enumerate(self.names):
            Lk = losses.get(k)
            if Lk is None:
                continue
            s = self.log_sigma[i]
            total = total + torch.exp(-s) * Lk + s
            parts[f"w_{k}"] = float(torch.exp(-s).detach())
        return total, parts
