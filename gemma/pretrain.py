"""gemma.pretrain -- SSL pretraining of the GEMMA v2 backbone on source cities."""
from __future__ import annotations

import numpy as np

import json
import math
import os
from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .backbone import GEMMA
from .config import Config
from .data import DiaryDataset, build_city_data, collate
from .fsq import FSQTaxonomy
from .poi_encoder import POIEncoder
from .utils import get_logger, set_seed


def pretrain(cfg: Config):
    set_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = get_logger("gemma.pretrain", os.path.join(run_dir, "pretrain.log"))
    device = cfg.resolved_device()
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    logger.info(f"device={device} pretrain_cities={cfg.pretrain_cities}")

    fsq = FSQTaxonomy(cfg.fsq_csv)
    if not os.path.exists(cfg.fsq_csv):
        raise FileNotFoundError(f"--fsq_csv not found: {cfg.fsq_csv} (this silently "
                                f"collapses the hierarchy to 1 class -- fix the path)")
    logger.info(f"FSQ taxonomy: {fsq.n_activity} activities, {fsq.n_primary} "
                f"primary, {fsq.n_secondary} secondary")
    assert fsq.n_activity >= 5, (
        f"taxonomy collapsed to {fsq.n_activity} activities -- check --fsq_csv path/format")
    enc = POIEncoder(cfg, logger)
    cfg.poi_emb_dim = enc.dim

    # auto-detect semantic-ID geometry from the built cache so the leaf head is
    # sized to match the codes (prevents the levels/codebook mismatch crash)
    if cfg.use_semantic_id:
        import glob
        npzs = glob.glob(os.path.join(cfg.sid_cache, "*_codes.npz"))
        if npzs:
            z = np.load(npzs[0], allow_pickle=True)
            det_levels = int(z["codes"].shape[1])
            det_K = int(z["codes"].max()) + 1
            if det_levels != cfg.sid_levels or det_K > cfg.sid_codebook:
                logger.info(f"semantic-ID cache -> levels={det_levels} "
                            f"max_code+1={det_K}; overriding config "
                            f"(was levels={cfg.sid_levels}, K={cfg.sid_codebook})")
            cfg.sid_levels = det_levels
            cfg.sid_codebook = max(cfg.sid_codebook, det_K)
        else:
            logger.warning(f"use_semantic_id=True but no *_codes.npz in "
                            f"{cfg.sid_cache} -> disabling; run build_semantic_ids.py")
            cfg.use_semantic_id = False

    cities, dsets = [], []
    for c in cfg.pretrain_cities:
        cd = build_city_data(cfg, c, cfg.train_splits, fsq, enc, logger)
        cities.append(cd); dsets.append(DiaryDataset(cd, cfg))
        logger.info(f"[{c}] {len(cd.df)} check-ins, {len(cd.trails)} trails, "
                    f"{len(dsets[-1])} diaries")
    cont_dim = cities[0].cont_dim
    # one collate over the concat set: each diary carries its own city rows already
    # (we store cont in the batch, so collate only needs a city for cont_dim/emb)
    # Simplest correct approach: separate loaders per city, round-robin.
    loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=use_cuda,
                          collate_fn=partial(collate, city=cd, cfg=cfg),
                          drop_last=len(ds) > cfg.batch_size)
               for ds, cd in zip(dsets, cities)]

    model = GEMMA(cfg, cont_dim, fsq.n_activity, fsq.n_primary, fsq.n_secondary).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params = {n_par/1e6:.2f}M | geometry={cfg.geometry} "
                f"masking={cfg.masking} reg={cfg.regularizer}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(l) for l in loaders)
    total = cfg.epochs * steps_per_epoch
    warm = int(cfg.warmup_frac * total)
    def lr_at(s):
        if s < warm:
            return s / max(1, warm)
        p = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    amp_on = cfg.amp and use_cuda
    amp_dtype = (torch.bfloat16 if amp_on and torch.cuda.is_bf16_supported() else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_on and amp_dtype == torch.float16))

    gstep = 0
    hist = open(os.path.join(run_dir, "history.jsonl"), "w")
    for epoch in range(cfg.epochs):
        model.train()
        iters = [iter(l) for l in loaders]
        agg = {}
        for it in range(steps_per_epoch):
            li = it % len(iters)
            try:
                batch = next(iters[li])
            except StopIteration:
                iters[li] = iter(loaders[li]); batch = next(iters[li])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_at(gstep)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                out = model(batch, use_reg=(epoch >= 1))
            if not torch.isfinite(out["loss"]):
                logger.warning(f"non-finite loss @ {gstep}; skip"); gstep += 1; continue
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            model.ema_update(cfg.ema_decay)
            rec = {k: (float(v.detach()) if hasattr(v, "detach") else float(v))
                   for k, v in out.items()}
            for k, v in rec.items():
                agg[k] = agg.get(k, 0.0) + v
            hist.write(json.dumps({"step": gstep, **{k: round(v, 4) for k, v in rec.items()}}) + "\n")
            if it % 50 == 0:
                logger.info(f"e{epoch} s{gstep} lr{cfg.lr*lr_at(gstep):.2e} "
                            f"loss={rec['loss']:.3f} pred={rec['pred']:.3f} sid={rec['sid']:.3f} "
                            f"geo={rec['geo']:.3f} tmp={rec['tmp']:.3f} motif={rec['motif']:.3f} | "
                            f"act@1={rec['act@1']:.3f} pri@1={rec['pri@1']:.3f} "
                            f"sec@1={rec['sec@1']:.3f} poi@1={rec['poi@1']:.3f}")
            gstep += 1
        n = max(1, steps_per_epoch)
        logger.info(f"== epoch {epoch} mean: " + " ".join(
            f"{k}={agg.get(k,0)/n:.4f}" for k in ["loss","pred","sid","geo","tmp","act@1","pri@1","sec@1","poi@1"]))
    hist.close()

    ckpt = os.path.join(run_dir, "gemma.pt")
    torch.save({"model": model.state_dict(), "cont_dim": cont_dim,
                "n_act": fsq.n_activity, "n_pri": fsq.n_primary,
                "n_sec": fsq.n_secondary, "config": cfg.__dict__}, ckpt)
    cfg.to_json(os.path.join(run_dir, "config.json"))
    logger.info(f"saved backbone -> {ckpt}")
    return model, fsq, enc, run_dir
