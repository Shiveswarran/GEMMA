"""Loading checkpoints and running one city in the zero-shot or fine-tuned setting."""
from __future__ import annotations

import os

import numpy as np
import torch

from .backbone import GEMMAEncoder, load_encoder_state
from .city import CityIndex, sthgcn_filter
from .data import build_city_data
from .evaluate import evaluate, evaluate_llm
from .model import new_head
from .train import train
from .utils import set_seed

SPLITS = ["train", "validation", "test"]

# published Massive-STEPS results of the strongest supervised model, STHGCN (Acc@1, NDCG@5)
STHGCN = {"bandung": (0.219, 0.302), "istanbul": (0.241, 0.318), "jakarta": (0.197, 0.270),
          "kuwait_city": (0.225, 0.314), "melbourne": (0.168, 0.247), "moscow": (0.223, 0.308),
          "new_york": (0.146, 0.207), "palembang": (0.246, 0.341), "petaling_jaya": (0.174, 0.241),
          "sao_paulo": (0.250, 0.344), "shanghai": (0.193, 0.264), "sydney": (0.227, 0.307),
          "tangerang": (0.293, 0.400), "tokyo": (0.250, 0.350)}


def load_encoder(cfg, encoder_dir, device, state=None, random_init=False):
    """Encoder architecture and pretrained weights from <encoder_dir>/gemma.pt.  ``state``
    (e.g. the encoder of a source-trained model) replaces the pretrained weights."""
    ck = torch.load(os.path.join(encoder_dir, "gemma.pt"), map_location="cpu", weights_only=False)
    enc = GEMMAEncoder(cfg, ck["cont_dim"], ck["n_act"], ck["n_pri"], ck["n_sec"]).to(device)
    if state is not None:
        load_encoder_state(enc, state)
    elif not random_init:
        load_encoder_state(enc, ck["model"])
    enc.eval()
    return enc


def load_source(path):
    """Source-trained model: {'head': state, 'encoder': state}."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "encoder" not in ck and "backbone" in ck:
        ck["encoder"] = ck["backbone"]
    return ck


def split_ids(city):
    ids = {s: [] for s in SPLITS}
    for i, t in enumerate(city.trails):
        ids.setdefault(t["split"], []).append(i)
    return ids


def run_city(cfg, run_dir, tax, text_enc, city_name, settings, protocols, logger, source,
             device):
    """settings: subset of {zero_shot, finetuned}; protocols: subset of {full, llm}.
    source: the source-trained checkpoint (dict with 'head' and 'encoder'), or None for the
    hand-set ablation (initial evidence weights, no content embedding)."""
    set_seed(cfg.seed)
    city = build_city_data(cfg, city_name, SPLITS, tax, text_enc, logger, filter_users=False)
    ids = split_ids(city)
    test_ids = ids[cfg.eval_split]
    filt = ((lambda ci_, q_: sthgcn_filter(ci_, q_, ids["train"])) if cfg.eval_protocol == "sthgcn"
            else (lambda ci_, q_: q_))
    rows = []
    for setting in settings:
        enc = load_encoder(cfg, run_dir, device, source["encoder"] if source else None)
        if setting == "zero_shot":
            # user histories from the training split; no population statistics, no update
            ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device=device)
            head = new_head(enc, ci, cfg, device)
            if source:
                head.load_state_dict(source["head"], strict=False)
            else:
                head.use_emb = False
        else:
            vis = ids["train"] if cfg.full_visible == "train" else ids["train"] + ids["validation"]
            ci = CityIndex(city, observed=vis, stat=vis, cfg=cfg, device=device)
            head = new_head(enc, ci, cfg, device)
            if source:
                head.load_state_dict(source["head"], strict=False)
            head.attach_city(ci.P, ci.n_users, device)
            rng = np.random.RandomState(cfg.seed)
            q_va = filt(ci, ci.queries(ids["validation"], False))
            q_va = [q_va[i] for i in rng.permutation(len(q_va))[:cfg.val_queries]]
            q_tr = ci.queries(ids["train"], cfg.ft_all_positions)
            logger.info(f"[{city_name}] fine-tuning on {len(q_tr)} next-step queries")
            train(enc, head, [(ci, q_tr)], cfg, device, logger, cfg.ft_epochs, cfg.ft_lr_head,
                  cfg.ft_lr_backbone, val=[(ci, q_va)] if q_va else None, tag=f"{city_name}/{setting}")
        q_all = ci.queries(test_ids, False)
        q_te = filt(ci, q_all)
        met = {"city": city_name, "setting": setting, "protocol": cfg.eval_protocol,
               "n_test_all": len(q_all), "n_test_scored": len(q_te)}
        if "full" in protocols:
            met.update(evaluate(enc, head, ci, q_te, cfg))
        if "llm" in protocols:
            met.update(evaluate_llm(enc, head, ci, test_ids, cfg))
        msg = f"[{city_name}/{setting}]"
        if "acc@1" in met:
            msg += (f" Acc@1={met['acc@1']:.4f} Acc@5={met['acc@5']:.4f} NDCG@5={met['ndcg@5']:.4f}"
                    f" | return share={met.get('return_share', float('nan')):.2f}"
                    f" explore@1 (within channel)={met.get('explore_cond_acc@1', float('nan')):.3f}")
            if city_name in STHGCN:
                msg += f" | STHGCN {STHGCN[city_name][0]:.3f}/{STHGCN[city_name][1]:.3f}"
        if "llm_acc@1" in met:
            msg += f" | LLM protocol Acc@1={met['llm_acc@1']:.3f}+-{met['llm_acc@1_std']:.3f}"
        logger.info(msg)
        rk = met.pop("_ranks", None)
        tag = f"{city_name}_{setting}{cfg.out_tag}"
        if rk is not None:
            np.save(os.path.join(run_dir, f"ranks_{tag}.npy"), rk)
        rows.append(met)
        del enc, head, ci
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows
