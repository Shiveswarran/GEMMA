"""Training loop shared by source training (several cities) and fine-tuning (one city).

The mixture head, and the encoder when ``lr_enc > 0``, are trained on every next-step
position of the given trajectories with AdamW and a cosine schedule.  Model selection uses
validation Acc@1: the initialization is the first candidate, validation runs every
``cfg.val_every`` steps and at the end of every epoch, and training stops after
``cfg.patience`` validations without improvement.  The best state is restored at the end.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from .evaluate import evaluate
from .model import forward, loss_fn
from .utils import amp_dtype

# encoder modules that only serve the pretraining objective and stay frozen afterwards
_FROZEN = ("embed_t", "enc_t", "motif", "weighter", "geo_head", "temporal_head")


def train(enc, head, jobs, cfg, device, logger, epochs, lr_head, lr_enc, val=None, tag=""):
    """jobs: list of (CityIndex, queries); batches of different cities are interleaved.
    val: list of (CityIndex, queries) used for model selection."""
    train_enc = lr_enc > 0
    for p in enc.parameters():
        p.requires_grad_(False)
    enc_params = []
    if train_enc:
        for name, p in enc.named_parameters():
            if not name.startswith(_FROZEN):
                p.requires_grad_(True); enc_params.append(p)
    resid = [p for n_, p in head.named_parameters() if n_.startswith(("poi_res", "user_res"))]
    rid = {id(p) for p in resid}
    core = [p for p in head.parameters() if p.requires_grad and id(p) not in rid]
    groups = [{"params": core, "lr": lr_head, "weight_decay": 1e-4, "base": lr_head}]
    if resid:
        groups.append({"params": resid, "lr": cfg.ft_lr_resid, "weight_decay": 0.05,
                       "base": cfg.ft_lr_resid})
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_enc, "weight_decay": 0.01, "base": lr_enc})
    opt = torch.optim.AdamW(groups)
    adt = amp_dtype(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(adt == torch.float16))
    rng = np.random.RandomState(cfg.seed)
    steps_ep = sum(math.ceil(min(len(q), cfg.ft_max_queries) / cfg.ft_batch) for _, q in jobs)
    total, step = max(1, epochs * steps_ep), 0
    logger.info(f"[{tag}] steps/epoch={steps_ep} epochs={epochs} lr_head={lr_head} lr_enc={lr_enc} "
                f"amp={adt} val_every={cfg.val_every or 'epoch'} patience={cfg.patience}")
    state = {"best": -1.0, "best_state": None, "bad": 0}

    def set_mode():
        enc.train() if train_enc else enc.eval()
        head.train()

    def validate(where):
        accs = [evaluate(enc, head, vci, vq, cfg) for vci, vq in val]
        acc = float(np.mean([a["acc@1"] for a in accs]))
        improved = acc > state["best"]
        if improved:
            state.update(best=acc, bad=0, best_state=(
                copy.deepcopy({k: v.cpu() for k, v in enc.state_dict().items()}),
                copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()})))
        else:
            state["bad"] += 1
        logger.info(f"[{tag}] val@{where}: acc@1={acc:.4f}{'  *best' if improved else ''}")
        set_mode()
        return state["bad"] >= cfg.patience

    if val:
        validate("init")
    stop = False
    for ep in range(epochs):
        set_mode()
        order = []
        for k, (_, q) in enumerate(jobs):
            idx = rng.permutation(len(q))[:cfg.ft_max_queries]
            order += [(k, idx[s:s + cfg.ft_batch]) for s in range(0, len(idx), cfg.ft_batch)]
        rng.shuffle(order)
        t0, tot, n = time.time(), 0.0, 0
        for it, (k, bidx) in enumerate(order):
            ci, q = jobs[k]
            for g in opt.param_groups:
                g["lr"] = g["base"] * 0.5 * (1 + math.cos(math.pi * step / total))
            out = forward(enc, head, ci, [q[i] for i in bidx], adt)
            loss = loss_fn(enc, out, ci, train_enc)
            if not torch.isfinite(loss):
                step += 1; continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); step += 1
            tot += float(loss.detach()); n += 1
            if (it + 1) % cfg.log_every == 0:
                el = time.time() - t0
                logger.info(f"[{tag}] ep{ep} step {it + 1}/{len(order)} loss={tot / n:.4f} "
                            f"(epoch ETA {el / (it + 1) * (len(order) - it - 1) / 60:.1f} min)")
            if val and cfg.val_every and (it + 1) % cfg.val_every == 0 and validate(f"ep{ep}-step{it + 1}"):
                stop = True; break
        logger.info(f"[{tag}] epoch {ep} loss={tot / max(1, n):.4f} ({time.time() - t0:.0f}s) | "
                    + head.describe())
        if stop or (val and validate(f"ep{ep}-end")):
            logger.info(f"[{tag}] early stop in epoch {ep}")
            break
    if state["best_state"] is not None:
        enc.load_state_dict(state["best_state"][0]); head.load_state_dict(state["best_state"][1])
        logger.info(f"[{tag}] restored best state (val acc@1={state['best']:.4f})")
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval(); head.eval()
    return head
