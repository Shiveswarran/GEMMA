"""
run_experiment.py
-----------------
End-to-end GEMMA experiment: pretrain jointly on the source cities, then
transfer to the target cities under zero-shot and few-shot retrieval.

Examples
--------
# quick 1-source -> 2-target glimpse
python run_experiment.py \
    --pretrain new_york \
    --transfer tokyo istanbul \
    --epochs 10 --run_name glimpse

# paper setup: pretrain on the five >90k-trail cities, transfer to the rest
python run_experiment.py \
    --pretrain istanbul petaling_jaya jakarta new_york kuwait_city \
    --transfer bandung beijing melbourne moscow palembang sao_paulo \
               shanghai sydney tangerang tokyo \
    --epochs 30 --run_name full --modes zero_shot few_shot
"""
from __future__ import annotations

import argparse
import json
import os

from gemma.config import GemmaConfig
from gemma.pretrain import pretrain
from gemma.data import build_city_data
from gemma.transfer import (build_source_pool, build_few_shot_pool,
                            evaluate_city)
from gemma.utils import get_logger, timer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--pretrain", nargs="+", default=["new_york"])
    p.add_argument("--transfer", nargs="+", default=["tokyo"])
    p.add_argument("--modes", nargs="+", default=["zero_shot", "few_shot"],
                   choices=["zero_shot", "few_shot"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_activities", type=int, default=64)
    p.add_argument("--name_embedder", default="sbert", choices=["sbert", "hash"])
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--run_name", default="gemma_run")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = GemmaConfig(
        data_root=a.data_root, pretrain_cities=a.pretrain,
        transfer_cities=a.transfer, epochs=a.epochs, batch_size=a.batch_size,
        d_model=a.d_model, n_activities=a.n_activities,
        name_embedder=a.name_embedder, device=a.device, out_dir=a.out_dir,
        run_name=a.run_name, seed=a.seed)

    # -- pretrain -----------------------------------------------------------
    model, vocab, embedder, run_dir = pretrain(cfg)
    device = cfg.resolved_device()
    logger = get_logger("gemma.transfer", os.path.join(run_dir, "transfer.log"))

    # -- source retrieval pool (from source train data already featurised) --
    with timer(logger, "build source retrieval pool"):
        source_cities = [build_city_data(cfg.data_root, c, "train", vocab,
                                         embedder, cfg, logger)
                         for c in cfg.pretrain_cities]
        source_pool = build_source_pool(model, source_cities, cfg, device, logger)

    # -- transfer + evaluate on each target city ----------------------------
    model.eval()
    summary = []
    for city in cfg.transfer_cities:
        with timer(logger, f"evaluate {city}"):
            target_test = build_city_data(cfg.data_root, city, "test", vocab,
                                          embedder, cfg, logger,
                                          filter_users=False)
            few_pool = None
            if "few_shot" in a.modes:
                target_train = build_city_data(cfg.data_root, city, "train",
                                               vocab, embedder, cfg, logger,
                                               filter_users=False)
                few_pool = build_few_shot_pool(model, target_train, cfg,
                                               device, logger)
            for mode in a.modes:
                m = evaluate_city(model, target_test, source_pool, cfg, device,
                                  logger, mode=mode, few_shot_pool=few_pool,
                                  run_dir=run_dir)
                summary.append(m)

    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("=== SUMMARY ===")
    for m in summary:
        logger.info(f"{m['city']:>12} {m['mode']:>10} | "
                    f"acc@1={m['acc@1']:.4f} acc@5={m['acc@5']:.4f} "
                    f"acc@10={m['acc@10']:.4f} ndcg@10={m['ndcg@10']:.4f} "
                    f"mrr={m['mrr']:.4f}")
    logger.info(f"all artifacts in {run_dir}")


if __name__ == "__main__":
    main()
