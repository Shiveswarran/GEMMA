"""GEMMA v5 downstream: zero/few/full-shot next-POI + next-activity on Massive-STEPS.

python scripts/run_downstream.py --run_dir runs/gemma_v5 --data_root data \
    --cities tokyo istanbul ... --shots zero_shot few_shot full --protocols full llm200
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.gpu_env import pin_gpu; pin_gpu()      # before torch is imported
import torch
from gemma.downstream import load_config, run_city, summary_table
from gemma.fsq import FSQTaxonomy
from gemma.poi_encoder import POIEncoder
from gemma.utils import check_device, get_logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True); p.add_argument("--data_root")
    p.add_argument("--fsq_csv")
    p.add_argument("--cities", nargs="+", required=True)
    p.add_argument("--shots", nargs="+", default=["zero_shot", "few_shot", "full"],
                   choices=["zero_shot", "few_shot", "full"])
    p.add_argument("--protocols", nargs="+", default=["full", "llm200"], choices=["full", "llm200"])
    p.add_argument("--zs_ckpt", default=None, help="default: <run_dir>/zs_realizer.pt if present")
    p.add_argument("--no_init_from_zs", action="store_true")
    p.add_argument("--device")
    # realization knobs (override config)
    for k, t in [("hist_trails", int), ("hist_max_tokens", int), ("few_shot_k", int),
                 ("ft_epochs", int), ("ft_epochs_few", int), ("ft_lr_head", float),
                 ("ft_lr_backbone", float), ("ft_batch", int), ("ft_max_queries", int),
                 ("rerank_m", int), ("val_queries", int), ("seed", int),
                 ("few_shot_frac", float)]:
        p.add_argument("--" + k, type=t)
    p.add_argument("--causal_history", action="store_true")
    p.add_argument("--no_history", action="store_true", help="ablation: prefix-only context")
    p.add_argument("--random_backbone", action="store_true", help="ablation: no pretraining")
    p.add_argument("--last_position_only", action="store_true")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--val_every", type=int); p.add_argument("--patience", type=int)
    p.add_argument("--ft_lr_resid", type=float)
    p.add_argument("--retriever", choices=["wm", "hrem", "realize"])
    for k, t in [("wm_H", int), ("wm_infill", float), ("lambda_wm_jepa", float),
                 ("lambda_wm_infill", float), ("lambda_wm_lat", float), ("lambda_wm_ptr", float),
                 ("min_epochs", int), ("hist_budget", int), ("hist_last", int), ("evidence", str), ("out_tag", str),
                 ("eval_protocol", str), ("eval_split", str), ("lambda_wm_motif", float),
                 ("lambda_wm_dt", float), ("drop_ret", str)]:
        p.add_argument("--" + k, type=t)
    a = p.parse_args()

    cfg = load_config(a.run_dir)
    for k, v in vars(a).items():
        if v is not None and hasattr(cfg, k) and k not in ("device",):
            setattr(cfg, k, v)
    if a.device: cfg.device = a.device
    if a.causal_history: cfg.causal_history = True
    if a.no_history: cfg.use_history = False
    if a.random_backbone: cfg.random_backbone = True
    if a.last_position_only: cfg.ft_all_positions = False
    if a.allow_cpu: cfg.allow_cpu = True
    logger = get_logger("gemma.down", os.path.join(a.run_dir, f"downstream_{cfg.retriever}.log"))
    device = check_device(cfg, logger)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    logger.info(f"retriever={cfg.retriever} predictor={cfg.predictor_type} hist_trails={cfg.hist_trails} "
                f"few_shot_k={cfg.few_shot_k} few_shot_frac={cfg.few_shot_frac} lr_bb={cfg.ft_lr_backbone} rerank_m={cfg.rerank_m}")
    fsq = FSQTaxonomy(cfg.fsq_csv); enc = POIEncoder(cfg, logger); cfg.poi_emb_dim = enc.dim

    zs_path = a.zs_ckpt or os.path.join(a.run_dir, "zs_realizer.pt" if cfg.retriever == "realize"
                                        else f"zs_{cfg.retriever}.pt")
    zs = torch.load(zs_path, map_location="cpu", weights_only=False) if os.path.exists(zs_path) else None
    if zs is not None and zs.get("kind", "realize") != cfg.retriever:
        logger.warning(f"{zs_path} was trained for retriever={zs.get('kind', 'realize')} "
                       f"but --retriever={cfg.retriever}; ignoring it")
        zs = None
    if zs is not None:                       # architecture knobs must match the pretrained model
        for k, v in (zs.get("cfg_overrides") or {}).items():
            # same objective at fine-tune time as at pretraining (ablations stay clean),
            # unless overridden on the command line
            if (k.startswith(("wm_", "lambda_wm_")) or k in ("evidence", "ema_decay")) and getattr(a, k, None) is None:
                setattr(cfg, k, v)
    logger.info(f"source-city realizer: {zs_path if zs else 'NONE'}")
    logger.info(f"evidence={getattr(cfg, 'evidence', 'full')} drop_ret={getattr(cfg, 'drop_ret', '')!r} hist_budget={getattr(cfg, 'hist_budget', -1)} tag={cfg.out_tag!r}")

    rows = []
    for city in a.cities:
        rows += run_city(cfg, a.run_dir, fsq, enc, city, a.shots, a.protocols, logger, zs,
                         device, init_from_zs=not a.no_init_from_zs)
        with open(os.path.join(a.run_dir, f"benchmark_summary_{cfg.retriever}{cfg.out_tag or ''}.json"), "w") as f:
            json.dump(rows, f, indent=2)
    summary_table(rows, logger)


if __name__ == "__main__":
    main()
