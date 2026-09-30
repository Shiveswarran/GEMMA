"""
gemma.downstream (v5) -- zero / few / full-shot evaluation with GEMMA-R.

Shots (what the model may use from the TARGET city):
  zero_shot : no target labels, no target statistics, no parameter update.
              Backbone (+ realization head post-trained on SOURCE cities).
              The user's own train-split history is inference-time context,
              exactly what the LLM zero-shot baselines receive.
  few_shot  : user history is inference context exactly as in zero-shot, but
              only <= few_shot_k LABELLED train trajectories per user are used
              for gradient updates and target statistics (10% of them held out
              for early stopping; the zero-shot initialisation is the epoch -1
              candidate, so adaptation can never end below zero-shot on val).
  full      : train+validation visible; fine-tune on train, early-stop on
              validation (standard in-city supervised protocol).

Protocols (reported side by side):
  full   : every test trajectory, rank over ALL city POIs (supervised tables)
  llm200 : 200 users x 1 trajectory, 100 nearest candidates, mean+-std over
           5 seeds (zero-shot LLM table)

Both tasks come from the SAME adapted model: next-POI and next-activity.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch

from .backbone import GEMMA
from .config import Config
from .data import build_city_data
from .realize import (CityIndex, evaluate, evaluate_llm_protocol, finetune, new_head)
from .sota_ref import SUP, ZS
from .utils import get_logger, set_seed

ALL_SPLITS = ["train", "validation", "test"]


def load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    raw = json.load(open(path))
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    if "predictor_type" not in raw:          # v4 checkpoints: pooled predictor
        cfg.predictor_type = "pooled"
    return cfg


def load_backbone(cfg, run_dir, device, state=None):
    ck = torch.load(os.path.join(run_dir, "gemma.pt"), map_location="cpu", weights_only=False)
    bb = GEMMA(cfg, ck["cont_dim"], ck["n_act"], ck["n_pri"], ck["n_sec"]).to(device)
    if state is not None:                 # e.g. source post-trained backbone
        bb.load_state_dict(state)
    elif not getattr(cfg, "random_backbone", False):
        bb.load_state_dict(ck["model"])
    # else: ablation -- same architecture, NO pretraining (does pretraining matter?)
    bb.eval()
    return bb


def split_ids(city):
    ids = {s: [] for s in ALL_SPLITS}
    for i, t in enumerate(city.trails):
        ids.setdefault(t["split"], []).append(i)
    return ids


def few_shot_subset(city, train_ids, k, seed, frac=0.0):
    rng = np.random.RandomState(seed)
    if frac and frac > 0:                       # budget = fraction of labelled trajectories
        n = max(2, int(round(frac * len(train_ids))))
        return sorted(int(i) for i in rng.choice(train_ids, min(n, len(train_ids)), replace=False))
    by_u = {}
    for i in train_ids:
        by_u.setdefault(city.trails[i]["user_id"], []).append(i)
    out = []
    for u, ts in by_u.items():
        out += list(rng.choice(ts, min(k, len(ts)), replace=False))
    return sorted(int(i) for i in out)


def run_city(cfg, run_dir, fsq, enc, city_name, shots, protocols, logger, zs=None,
             device="cuda", init_from_zs=True):
    set_seed(cfg.seed)
    city = build_city_data(cfg, city_name, ALL_SPLITS, fsq, enc, logger, filter_users=False)
    ids = split_ids(city)
    test_ids = ids["test"]
    logger.info(f"[{city_name}] trails train={len(ids['train'])} val={len(ids['validation'])} "
                f"test={len(test_ids)} | POIs (all splits) = {len(city.poi_table['venue_ids'])}")
    source = city_name in cfg.pretrain_cities
    rows = []
    for shot in shots:
        bb_state = zs["backbone"] if (zs and zs.get("backbone") is not None and
                                      (shot == "zero_shot" or init_from_zs)) else None
        bb = load_backbone(cfg, run_dir, device, bb_state)
        if shot == "zero_shot":
            ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device=device)
            head = new_head(bb, ci, cfg, device)
            if zs:
                head.load_state_dict(zs["head"], strict=False)
            else:
                head.use_emb = False          # untrained embedding term would be noise
                logger.warning(f"[{city_name}] no zs_head.pt -> hand-initialised evidence "
                               f"fusion only (run scripts/train_realizer_source.py)")
        else:
            # Context vs labels.  The user's train history is INFERENCE CONTEXT in
            # every shot (zero-shot already sees it), so few-shot must see it too;
            # what 'few' restricts is the LABELLED data used for gradient updates
            # and for target-city statistics (covis / pop / hour).
            if shot == "few_shot":
                few = few_shot_subset(city, ids["train"], cfg.few_shot_k, cfg.seed,
                                      getattr(cfg, "few_shot_frac", 0.0))
                rng = np.random.RandomState(cfg.seed)
                perm = rng.permutation(len(few))
                n_va = max(1, int(0.1 * len(few)))
                few_va = sorted(few[i] for i in perm[:n_va])
                few_tr = sorted(few[i] for i in perm[n_va:])
                ci = CityIndex(city, observed=ids["train"], stat=few, cfg=cfg, device=device)
                train_ids, vis = few_tr, few
                q_va = ci.queries(few_va, False)
                val, ep = ((ci, q_va) if q_va else None), cfg.ft_epochs_few
            else:
                vis = ids["train"] + ids["validation"]
                ci = CityIndex(city, observed=vis, stat=vis, cfg=cfg, device=device)
                train_ids = ids["train"]
                rng = np.random.RandomState(cfg.seed)
                q_va = ci.queries(ids["validation"], False)
                q_va = [q_va[i] for i in rng.permutation(len(q_va))[:cfg.val_queries]]
                val, ep = ((ci, q_va) if q_va else None), cfg.ft_epochs
            head = new_head(bb, ci, cfg, device)
            if zs and init_from_zs:
                head.load_state_dict(zs["head"], strict=False)
            head.attach_city(ci.P, ci.n_users, device)
            q_tr = ci.queries(train_ids, cfg.ft_all_positions)
            logger.info(f"[{city_name}/{shot}] fine-tune on {len(q_tr)} next-step queries "
                        f"from {len(train_ids)} trajectories (visible={len(vis)})")
            finetune(bb, head, [(ci, q_tr)], cfg, device, logger, ep,
                     cfg.ft_lr_head, cfg.ft_lr_backbone, val=val, tag=f"{city_name}/{shot}")
        q_te = ci.queries(test_ids, False)
        met = {"city": city_name, "shot": shot, "source_city": source}
        if "full" in protocols:
            met.update(evaluate(bb, head, ci, q_te, cfg))
        if "llm200" in protocols:
            met.update(evaluate_llm_protocol(bb, head, ci, test_ids, cfg))
        sup, zsr = SUP.get(city_name), ZS.get(city_name)
        msg = f"[{city_name}/{shot}{' (source)' if source else ''}] "
        if "acc@1" in met:
            msg += (f"FULL-VOCAB acc@1={met['acc@1']:.4f} acc@5={met['acc@5']:.4f} "
                    f"acc@10={met['acc@10']:.4f} ndcg@5={met['ndcg@5']:.4f} mrr={met['mrr']:.4f} "
                    f"| next-activity acc={met['act@1']:.4f} f1={met['act_macro_f1']:.4f}")
            if "revisit_acc@1" in met:
                msg += (f" | revisit({met['revisit_frac']:.2f}) acc@1={met['revisit_acc@1']:.3f}"
                        f" explore acc@1={met.get('explore_acc@1', float('nan')):.3f}")
            if sup:
                msg += f" | STHGCN {sup[0]:.3f}/{sup[1]:.3f}"
        if "llm_acc@1" in met:
            msg += (f" || LLM-200[gt+near100] acc@1={met['llm_acc@1']:.3f}+-{met['llm_acc@1_std']:.3f} "
                    f"acc@5={met['llm_acc@5']:.3f} ndcg@5={met['llm_ndcg@5']:.3f}"
                    f" [near100 {met['llm_near100_acc@1']:.3f}] [gt+rand100 {met['llm_gtrand100_acc@1']:.3f}]")
            if zsr:
                msg += f" | LLM-Move best {zsr[0]:.3f}/{zsr[1]:.3f}"
        logger.info(msg)
        with open(os.path.join(run_dir, f"v5_{city_name}_{shot}.json"), "w") as f:
            json.dump(met, f, indent=2)
        rows.append(met)
        del bb, head, ci
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def summary_table(rows, logger):
    logger.info("=== GEMMA v5 SUMMARY (A@1 / N@5; ref = STHGCN for few/full, LLM-Move for zero-shot) ===")
    logger.info(f"{'city':>14} {'shot':<9} {'full-vocab':>15} {'LLM-200':>15} {'ref':>13} "
                f"{'next-act':>8}")
    for m in rows:
        fv = f"{m.get('acc@1', float('nan')):.3f}/{m.get('ndcg@5', float('nan')):.3f}"
        lp = f"{m.get('llm_acc@1', float('nan')):.3f}/{m.get('llm_ndcg@5', float('nan')):.3f}"
        ref = (ZS if m["shot"] == "zero_shot" else SUP).get(m["city"], (float("nan"),) * 2)
        logger.info(f"{m['city']:>14} {m['shot']:<9} {fv:>15} {lp:>15} "
                    f"{ref[0]:.3f}/{ref[1]:.3f} {m.get('act@1', float('nan')):>8.3f}"
                    f"{'  *source' if m.get('source_city') else ''}")


class ExternalAdapter:
    """Contract for wrapping external SOTA repos (GETNext/STHGCN/...): export a
    per-POI GEMMA realization embedding aligned to venue ids."""
    def __init__(self, backbone, head, cfg, device):
        self.bb, self.head, self.cfg, self.device = backbone, head, cfg, device

    @torch.no_grad()
    def export(self, ci):
        E = self.head.poi_matrix(ci).cpu().numpy()
        return {"poi_feat": E, "poi_ids": ci.city.poi_table["venue_ids"]}
