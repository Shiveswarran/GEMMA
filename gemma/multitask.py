"""
gemma.multitask -- uncertainty weighting with FLOORS + fixed-weight core tasks.

Kendall weighting sets w_k ~ exp(-s_k) ~ 1/(2 sigma_k^2), which DOWN-weights
high-variance tasks.  The semantic-ID (code generation) head is inherently the
noisiest task, so vanilla weighting starves it (observed: w_sid 1.0 -> 0.35 while
w_jepa -> 5.1, and sid loss stopped falling).  Two guards:

  * fixed_tasks: given a constant weight, EXCLUDED from uncertainty weighting
    (use for the core objectives you never want throttled: 'sid', optionally 'jepa').
  * min_w: a floor on the learned weight of every uncertainty-weighted task, so
    none can be driven toward zero.

L = sum_{k in fixed}   c_k * L_k
  + sum_{k in learned}  max(exp(-s_k), min_w) * L_k + s_k
"""
import torch
import torch.nn as nn


class UncertaintyWeighter(nn.Module):
    def __init__(self, task_names, fixed=None, min_w=0.25):
        super().__init__()
        self.names = list(task_names)
        self.fixed = dict(fixed or {})                      # {task: constant weight}
        self.min_w = float(min_w)
        learned = [n for n in self.names if n not in self.fixed]
        self.learned = learned
        self.log_sigma = nn.Parameter(torch.zeros(len(learned)))

    def forward(self, losses: dict):
        total = 0.0
        parts = {}
        # fixed-weight core tasks (never throttled)
        for k, c in self.fixed.items():
            Lk = losses.get(k)
            if Lk is not None:
                total = total + c * Lk
                parts[f"w_{k}"] = float(c)
        # uncertainty-weighted auxiliary tasks, with a floor
        for i, k in enumerate(self.learned):
            Lk = losses.get(k)
            if Lk is None:
                continue
            w = torch.exp(-self.log_sigma[i])
            w_floored = torch.clamp(w, min=self.min_w)
            total = total + w_floored * Lk + self.log_sigma[i]
            parts[f"w_{k}"] = float(w_floored.detach())
        return total, parts
