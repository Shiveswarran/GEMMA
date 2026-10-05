"""Stage-2 (source-city) post-training of the GEMMA realizer -> zero-shot model.

Trains the realization head (and, with --tune_backbone, the backbone at a small
LR) on the SOURCE (pretraining) cities' train splits with target statistics
switched OFF, i.e. in exactly the zero-shot input regime (backbone + user's own
history + POI metadata/geography).  No target-city data is touched, so applying
the result to a held-out city is genuine zero-shot transfer -- the analogue of
instruction-tuning a foundation model before zero-shot use.

python scripts/train_realizer_source.py --run_dir runs/gemma_v5 --data_root data \
    --cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo --tune_backbone
-> <run_dir>/zs_realizer.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.gpu_env import pin_gpu; pin_gpu()      # before torch is imported
import numpy as np, torch
from gemma.data import build_city_data
from gemma.downstream import load_backbone, load_config, split_ids
from gemma.fsq import FSQTaxonomy
from gemma.poi_encoder import POIEncoder
from gemma.realize import CityIndex, finetune, new_head
from gemma.utils import check_device, get_logger, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True); p.add_argument("--data_root")
    p.add_argument("--cities", nargs="+", default=None, help="default: config.pretrain_cities")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max_queries_per_city", type=int, default=150000)
    p.add_argument("--tune_backbone", action="store_true")
    p.add_argument("--lr_head", type=float, default=2e-3)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device"); p.add_argument("--out", default=None)
    p.add_argument("--random_backbone", action="store_true", help="ablation: no pretraining")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--retriever", choices=["wm", "hrem", "realize"], default=None)
    for k, t in [("wm_H", int), ("wm_infill", float), ("wm_pred_layers", int),
                 ("lambda_wm_jepa", float), ("lambda_wm_infill", float), ("lambda_wm_probe", float),
                 ("lambda_wm_reg", float), ("lambda_wm_lat", float), ("lambda_wm_ptr", float),
                 ("seed", int), ("patience", int), ("min_epochs", int), ("log_every", int), ("evidence", str)]:
        p.add_argument("--" + k, type=t)
    p.add_argument("--wm_flat", action="store_true")
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_per_city", type=int, default=1000,
                   help="validation queries per source city (SE of acc@1 ~ 0.5/sqrt(6*n))")
    a = p.parse_args()
    cfg = load_config(a.run_dir)
    if a.data_root: cfg.data_root = a.data_root
    if a.device: cfg.device = a.device
    cfg.ft_batch = a.batch; cfg.ft_max_queries = a.max_queries_per_city
    cfg.random_backbone = a.random_backbone
    cfg.allow_cpu = a.allow_cpu; cfg.val_every = a.val_every
    if a.retriever: cfg.retriever = a.retriever
    for k in ("wm_H", "wm_infill", "wm_pred_layers", "lambda_wm_jepa", "lambda_wm_infill",
              "lambda_wm_probe", "lambda_wm_reg", "lambda_wm_lat", "lambda_wm_ptr", "seed",
              "patience", "min_epochs", "log_every", "evidence"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.wm_flat: cfg.wm_flat = True
    logger = get_logger("gemma.zs", os.path.join(a.run_dir, "train_realizer_source.log"))
    device = check_device(cfg, logger); set_seed(cfg.seed)
    fsq = FSQTaxonomy(cfg.fsq_csv); enc = POIEncoder(cfg, logger); cfg.poi_emb_dim = enc.dim
    bb = load_backbone(cfg, a.run_dir, device)
    jobs, vals = [], []
    for c in (a.cities or cfg.pretrain_cities):
        city = build_city_data(cfg, c, ["train", "validation"], fsq, enc, logger, filter_users=False)
        ids = split_ids(city)
        ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device=device)
        q = ci.queries(ids["train"], cfg.ft_all_positions)
        rng = np.random.RandomState(cfg.seed)
        qv = ci.queries(ids["validation"], False)
        qv = [qv[i] for i in rng.permutation(len(qv))[:a.val_per_city]]
        jobs.append((ci, q)); vals.append((ci, qv))
        logger.info(f"[{c}] source queries={len(q)} val={len(qv)} POIs={ci.P}")
    head = new_head(bb, jobs[0][0], cfg, device)
    finetune(bb, head, jobs, cfg, device, logger, a.epochs, a.lr_head,
             a.lr_backbone if a.tune_backbone else 0.0, val=vals, tag="source")
    out = a.out or os.path.join(a.run_dir, "zs_realizer.pt" if cfg.retriever == "realize"
                                else f"zs_{cfg.retriever}.pt")
    torch.save({"head": {k: v.cpu() for k, v in head.state_dict().items()},
                "backbone": ({k: v.cpu() for k, v in bb.state_dict().items()}
                             if a.tune_backbone else None),
                "cities": a.cities or cfg.pretrain_cities, "kind": cfg.retriever,
                "cfg_overrides": {k: getattr(cfg, k) for k in (
                    "wm_H", "wm_flat", "wm_infill", "wm_pred_layers", "lambda_wm_jepa",
                    "lambda_wm_infill", "lambda_wm_probe", "lambda_wm_reg", "lambda_wm_lat",
                    "lambda_wm_ptr", "retriever", "evidence")}}, out)
    logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
