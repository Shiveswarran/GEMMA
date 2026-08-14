"""
gemma.pretrain
--------------
Self-supervised pretraining on the source cities (Problem 3.1 in the paper).

Trains the context encoder, predictor, activity codebook, transition operator
and grammar head jointly by:
    L = lambda_pred * L_pred      (JEPA latent prediction)
      + lambda_act  * L_act       (next-activity grammar cross-entropy)
      + lambda_markov * L_markov  (Markov consistency with operator T)
      + lambda_sig  * L_sig       (SIGReg isotropic-Gaussian, warm-started)
      + lambda_cb   * L_cb        (codebook usage)

Saves: config.json, gemma.pt (weights), history.jsonl (per-step losses),
transition_matrix.npy, and vocab/embedder metadata for transfer.
"""
from __future__ import annotations

import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GemmaConfig
from .data import (build_category_vocab, build_city_data, PretrainDiaryDataset,
                   collate_windows)
from .features import NameEmbedder
from .model import GEMMA
from .utils import get_logger, set_seed, timer


def _make_embedder(cfg, logger):
    return NameEmbedder(backend=cfg.name_embedder, model_name=cfg.sbert_model,
                        dim=cfg.name_emb_dim, cache_dir=os.path.join(
                            cfg.out_dir, "cache_text"), logger=logger)


def pretrain(cfg: GemmaConfig):
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = get_logger("gemma.pretrain", os.path.join(run_dir, "pretrain.log"))
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # autotune conv/matmul
    amp_on = bool(cfg.amp) and use_cuda
    amp_dtype = (torch.bfloat16 if amp_on and torch.cuda.is_bf16_supported()
                 else torch.float16)
    # GradScaler is only needed for fp16; bf16 has fp32 range and skips it.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(amp_on and amp_dtype == torch.float16))
    logger.info(f"device={device} | pretrain_cities={cfg.pretrain_cities}")
    if use_cuda:
        logger.info(f"amp={amp_on} ({amp_dtype if amp_on else 'off'}) "
                    f"tf32=on cudnn.benchmark=on num_workers={cfg.num_workers}")
    cfg.to_json(os.path.join(run_dir, "config.json"))

    # -- shared category vocab over ALL cities (source + target) ------------
    all_cities = list(dict.fromkeys(cfg.pretrain_cities + cfg.transfer_cities))
    with timer(logger, "build global category vocab"):
        vocab = build_category_vocab(cfg.data_root, all_cities)
    logger.info(f"category vocab: {vocab.n_cat} categories, "
                f"{vocab.n_primary} primary")

    embedder = _make_embedder(cfg, logger)
    if cfg.coarse_activities:
        vocab.build_coarse(embedder, cfg.n_coarse_activities, logger)
        logger.info(f"grammar target space: {vocab.n_primary} coarse activities")

    # -- build source-city data --------------------------------------------
    cities = []
    with timer(logger, "load + featurise source cities"):
        for c in cfg.pretrain_cities:
            cities.append(build_city_data(cfg.data_root, c, "train",
                                          vocab, embedder, cfg, logger))
    cont_dim = cities[0].cont_dim
    logger.info(f"continuous feature dim = {cont_dim}")

    ds = PretrainDiaryDataset(cities, cfg, rng_seed=cfg.seed)
    logger.info(f"pretrain windows source (diaries usable) = {len(ds)}")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers,
                        pin_memory=(cfg.pin_memory and use_cuda),
                        persistent_workers=(cfg.num_workers > 0),
                        collate_fn=partial(collate_windows,
                                           block_max=cfg.mask_block_max),
                        drop_last=len(ds) > cfg.batch_size)

    # -- model / optim ------------------------------------------------------
    model = GEMMA(cfg, cont_dim, vocab.n_cat, vocab.n_primary).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params = {n_params/1e6:.2f}M")
    if cfg.compile and use_cuda:
        logger.info("torch.compile(model) — first step will be slow (tracing)")
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.epochs * len(loader))

    def lr_at(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    hist_path = os.path.join(run_dir, "history.jsonl")
    hist = open(hist_path, "w")
    gstep = 0
    for epoch in range(cfg.epochs):
        model.train()
        use_sig = epoch >= cfg.sigreg_warmup_epochs
        agg = {}
        for it, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_at(gstep)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_on):
                out = model(batch, use_sig=use_sig)
            if not torch.isfinite(out["loss"]):
                logger.warning(f"e{epoch} s{gstep} non-finite loss "
                               f"({float(out['loss'])}); skipping step")
                opt.zero_grad(set_to_none=True)
                gstep += 1
                continue
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.ema_update(cfg.ema_decay)

            rec = {k: float(v.detach()) for k, v in out.items()}
            rec.update({"epoch": epoch, "step": gstep, "lr": cfg.lr * lr_at(gstep)})
            hist.write(json.dumps(rec) + "\n")
            for k, v in rec.items():
                agg[k] = agg.get(k, 0.0) + v
            if gstep % cfg.log_every == 0:
                logger.info(
                    f"e{epoch} s{gstep} lr{cfg.lr*lr_at(gstep):.2e} "
                    f"loss={rec['loss']:.3f} pred={rec['pred']:.3f} "
                    f"cat={rec['cat']:.3f} act={rec['act']:.3f} "
                    f"sig={rec['sig']:.3f} | cat@1={rec['cat_top1']:.3f} "
                    f"act@1={rec['act_top1']:.3f}"
                    f"{'' if use_sig else '  [sig off: warmup]'}")
            gstep += 1
        n = max(1, it + 1)
        logger.info(f"== epoch {epoch} mean: "
                    + " ".join(f"{k}={agg[k]/n:.4f}"
                               for k in ["loss", "pred", "cat", "cat_top1",
                                         "act_top1"]))
    hist.close()

    # -- persist artifacts --------------------------------------------------
    ckpt = os.path.join(run_dir, "gemma.pt")
    torch.save({"model": model.state_dict(), "cont_dim": cont_dim,
                "n_cat": vocab.n_cat, "n_primary": vocab.n_primary,
                "cat2id": vocab.cat2id, "primary2id": vocab.primary2id,
                "config": cfg.__dict__}, ckpt)
    T = torch.softmax(model.codebook.trans.detach().cpu(), dim=-1).numpy()
    np.save(os.path.join(run_dir, "transition_matrix.npy"), T)
    logger.info(f"saved checkpoint -> {ckpt}")
    logger.info(f"saved transition matrix (K={T.shape[0]}) -> "
                f"{os.path.join(run_dir,'transition_matrix.npy')}")
    return model, vocab, embedder, run_dir


if __name__ == "__main__":
    pretrain(GemmaConfig())
