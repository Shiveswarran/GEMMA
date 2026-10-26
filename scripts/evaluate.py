"""Evaluate GEMMA on target cities: zero-shot and/or fine-tuned, supervised and/or LLM protocol.

# zero-shot and fine-tuned, supervised protocol (all places of the city are candidates)
python scripts/evaluate.py --run_dir runs/gemma --source runs/gemma/source_s0.pt \
    --cities tokyo shanghai ... --settings zero_shot finetuned --out_tag _s0

# LLM-Move protocol (true place + 100 random places, last 15 trajectories as history)
python scripts/evaluate.py --run_dir runs/gemma --source runs/gemma/source_s0.pt \
    --cities tokyo ... --settings zero_shot --protocols llm --hist_last 15 --eval_protocol all

Test-time ablations: --drop_ret hist,tod  --drop_exp hier  --gate_mode noS|nopersonal|const
                     --evidence none (no personal statistics)
                     --hist_budget k (only the k most recent trajectories before the query)
                     --source none (hand-set evidence weights, no learning)
-> <run_dir>/results_<tag>.json and per-query ranks <run_dir>/ranks_<city>_<setting><tag>.npy
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.pipeline import load_source, run_city  # noqa: E402
from gemma.taxonomy import Taxonomy  # noqa: E402
from gemma.text_encoder import TextEncoder  # noqa: E402
from gemma.utils import check_device, get_logger, load_config, save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--source", required=True, help="source-trained model, or 'none'")
    p.add_argument("--cities", nargs="+", required=True)
    p.add_argument("--settings", nargs="+", default=["zero_shot", "finetuned"],
                   choices=["zero_shot", "finetuned"])
    p.add_argument("--protocols", nargs="+", default=["full"], choices=["full", "llm"])
    p.add_argument("--full_visible", default="train", choices=["train", "train+val"],
                   help="fine-tuned setting: trajectories visible at test time (paper: train)")
    p.add_argument("--allow_cpu", action="store_true")
    for k, t in [("data_root", str), ("fsq_csv", str), ("device", str), ("seed", int),
                 ("hist_last", int), ("hist_budget", int), ("eval_protocol", str), ("eval_split", str),
                 ("drop_ret", str), ("drop_exp", str), ("gate_mode", str), ("evidence", str),
                 ("out_tag", str), ("ft_epochs", int), ("ft_batch", int), ("ft_lr_head", float),
                 ("ft_lr_backbone", float), ("val_queries", int)]:
        p.add_argument("--" + k, type=t)
    a = p.parse_args()
    cfg = load_config(a.run_dir)
    for k, v in vars(a).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    logger = get_logger("gemma.eval", os.path.join(a.run_dir, f"evaluate{cfg.out_tag}.log"))
    device = check_device(cfg, logger)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    tax, text_enc = Taxonomy(cfg.fsq_csv), TextEncoder(cfg, logger)
    cfg.poi_emb_dim = text_enc.dim
    source = None if a.source == "none" else load_source(a.source)
    logger.info(f"source model: {a.source} | protocol={cfg.eval_protocol} drop_ret={cfg.drop_ret!r} "
                f"drop_exp={cfg.drop_exp!r} gate_mode={cfg.gate_mode!r} hist_budget={cfg.hist_budget}")
    rows = []
    for city in a.cities:
        rows += run_city(cfg, a.run_dir, tax, text_enc, city, a.settings, a.protocols, logger,
                         source, device)
        save_json(rows, os.path.join(a.run_dir, f"results{cfg.out_tag}.json"))


if __name__ == "__main__":
    main()
