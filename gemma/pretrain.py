"""Self-supervised pretraining of the encoder on the source cities (see gemma/backbone.py)."""
from __future__ import annotations

import glob
import json
import math
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import GEMMAEncoder
from .data import DiaryDataset, build_city_data, collate
from .taxonomy import Taxonomy
from .text_encoder import TextEncoder
from .utils import amp_dtype, check_device, get_logger, set_seed


def pretrain(cfg):
    set_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = get_logger("gemma.pretrain", os.path.join(run_dir, "pretrain.log"))
    device = check_device(cfg, logger)
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(cfg.fsq_csv):
        raise FileNotFoundError(f"taxonomy file not found: {cfg.fsq_csv}")
    tax = Taxonomy(cfg.fsq_csv)
    logger.info(f"taxonomy: {tax.n_activity} activities, {tax.n_primary} primary, "
                f"{tax.n_secondary} leaf categories")
    text_enc = TextEncoder(cfg, logger)
    cfg.poi_emb_dim = text_enc.dim

    if cfg.use_semantic_id:                     # size the code head from the code cache
        npzs = glob.glob(os.path.join(cfg.sid_cache, "*_codes.npz"))
        if npzs:
            codes = np.load(npzs[0], allow_pickle=True)["codes"]
            cfg.sid_levels = int(codes.shape[1])
            cfg.sid_codebook = max(cfg.sid_codebook, int(codes.max()) + 1)
        else:
            logger.warning(f"no semantic codes in {cfg.sid_cache}; run scripts/build_semantic_ids.py")
            cfg.use_semantic_id = False

    cities, loaders = [], []
    for c in cfg.pretrain_cities:
        cd = build_city_data(cfg, c, cfg.train_splits, tax, text_enc, logger)
        ds = DiaryDataset(cd, cfg)
        cities.append(cd)
        loaders.append(DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=use_cuda,
                                  collate_fn=partial(collate, city=cd, cfg=cfg),
                                  drop_last=len(ds) > cfg.batch_size))
        logger.info(f"[{c}] {len(cd.df)} check-ins, {len(cd.trails)} trajectories, {len(ds)} diaries")
    cont_dim = cities[0].cont_dim

    model = GEMMAEncoder(cfg, cont_dim, tax.n_activity, tax.n_primary, tax.n_secondary).to(device)
    logger.info(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(l) for l in loaders)
    total = cfg.epochs * steps_per_epoch
    warm = int(cfg.warmup_frac * total)

    def lr_at(s):
        if s < warm:
            return s / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    adt = amp_dtype(cfg, device)
    amp_on = adt is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_on and adt == torch.float16))
    gstep = 0
    hist = open(os.path.join(run_dir, "history.jsonl"), "w")
    for epoch in range(cfg.epochs):
        model.train()
        iters = [iter(l) for l in loaders]
        agg = {}
        for it in range(steps_per_epoch):          # round-robin over cities
            li = it % len(iters)
            try:
                batch = next(iters[li])
            except StopIteration:
                iters[li] = iter(loaders[li]); batch = next(iters[li])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_at(gstep)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=adt or torch.float16, enabled=amp_on):
                out = model(batch, use_reg=(epoch >= 1))
            if not torch.isfinite(out["loss"]):
                logger.warning(f"non-finite loss at step {gstep}; skipped"); gstep += 1; continue
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            model.ema_update(cfg.ema_decay)
            rec = {k: float(v.detach()) if hasattr(v, "detach") else float(v) for k, v in out.items()}
            for k, v in rec.items():
                agg[k] = agg.get(k, 0.0) + v
            hist.write(json.dumps({"step": gstep, **{k: round(v, 4) for k, v in rec.items()}}) + "\n")
            if it % 50 == 0:
                logger.info(f"e{epoch} s{gstep} loss={rec['loss']:.3f} jepa={rec['pred']:.3f} "
                            f"sid={rec['sid']:.3f} act@1={rec['act@1']:.3f} sec@1={rec['sec@1']:.3f}")
            gstep += 1
        n = max(1, steps_per_epoch)
        logger.info(f"== epoch {epoch}: " + " ".join(
            f"{k}={agg.get(k, 0) / n:.4f}" for k in ("loss", "pred", "sid", "geo", "tmp", "act@1", "sec@1")))
    hist.close()
    torch.save({"model": model.state_dict(), "cont_dim": cont_dim, "n_act": tax.n_activity,
                "n_pri": tax.n_primary, "n_sec": tax.n_secondary}, os.path.join(run_dir, "gemma.pt"))
    cfg.to_json(os.path.join(run_dir, "config.json"))
    logger.info(f"saved encoder -> {run_dir}/gemma.pt")
