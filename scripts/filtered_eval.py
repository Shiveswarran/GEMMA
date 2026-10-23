"""Re-score saved per-query ranks on the SAME test subset the supervised baselines use.

The Massive-STEPS STHGCN fork (w11wo/Spatio-Temporal-Hypergraph-Model,
preprocess/preprocess_fn.py::remove_unseen_user_poi) DROPS every validation/test
check-in whose POI or user does not occur in the training split, and evaluates the
last remaining check-in of each test trajectory.  Our evaluation keeps every test
trajectory, including targets that were never visited in the training split (which
no ID-based model can predict).  This script reports both views from the ranks we
already saved -- no model is re-run.

python scripts/filtered_eval.py --run_dir runs/gemma_v4 --data_root data \
   --cities tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung \
   --ranks "GEMMA-zero=runs/gemma_v4/ranks_hrem_{city}_zero_shot_G0.npy" \
           "hand=runs/abl_incity/ranks_hrem_{city}_zero_shot_G0.npy" \
           "GEMMA+knn=runs/gemma_v4/ranks_mem_knn_{city}.npy"
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gemma.downstream import load_config, split_ids, ALL_SPLITS
from gemma.data import build_city_data
from gemma.fsq import FSQTaxonomy
from gemma.poi_encoder import POIEncoder
from gemma.realize import CityIndex
from gemma.metrics import ranking_metrics
from gemma.sota_ref import SUP
from gemma.utils import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_root")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--ranks", nargs="+", required=True, help='"label=path_with_{city}"')
    ap.add_argument("--out", default="logs/filtered_eval.json")
    a = ap.parse_args()
    cfg = load_config(a.run_dir)
    if a.data_root: cfg.data_root = a.data_root
    logger = get_logger("gemma.filt")
    fsq = FSQTaxonomy(cfg.fsq_csv); enc = POIEncoder(cfg, logger); cfg.poi_emb_dim = enc.dim
    specs = [s.split("=", 1) for s in a.ranks]
    rows = []
    pooled = {lab: {"all": [], "seen": []} for lab, _ in specs}
    for c in a.cities:
        city = build_city_data(cfg, c, ALL_SPLITS, fsq, enc, logger, filter_users=False)
        ids = split_ids(city)
        ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device="cpu")
        train_pois = set(int(p) for i in ids["train"] for p in ci.row_pi[city.trails[i]["rows"]])
        train_users = set(city.trails[i]["user_id"] for i in ids["train"])
        q = ci.queries(ids["test"], False)
        tgt = np.array([int(ci.row_pi[city.trails[ti]["rows"][j]]) for ti, j in q])
        usr = np.array([city.trails[ti]["user_id"] for ti, _ in q])
        seen = np.array([t in train_pois for t in tgt]) & np.array([u in train_users for u in usr])
        r = {"city": c, "n_all": int(len(q)), "n_seen": int(seen.sum()),
             "frac_unseen_target": float(1 - seen.mean())}
        msg = f"[{c}] test={len(q)} kept={seen.sum()} (unseen-target {1 - seen.mean():.1%})"
        for lab, pat in specs:
            f = pat.format(city=c)
            if not os.path.exists(f):
                msg += f" | {lab}: missing"; continue
            rk = np.load(f)
            if len(rk) != len(q):
                msg += f" | {lab}: length mismatch {len(rk)}!={len(q)}"; continue
            ma, ms = ranking_metrics(rk), ranking_metrics(rk[seen])
            r[lab] = {"all": ma, "seen": ms}
            pooled[lab]["all"].append(rk); pooled[lab]["seen"].append(rk[seen])
            # one-sample test vs the published STHGCN Acc@1 (treated as a constant; its own
            # run-to-run variance is unknown, so this understates the uncertainty)
            n_s = int(seen.sum()); p_s = ms["acc@1"]
            se = float(np.sqrt(max(p_s * (1 - p_s), 1e-9) / max(n_s, 1)))
            ref = SUP.get(c, (float("nan"),))[0]
            z = (p_s - ref) / se if se > 0 else float("nan")
            ms["se"], ms["z_vs_sthgcn"] = se, z
            flag = "  >" if z > 1.96 else ("  <" if z < -1.96 else "  =")
            msg += (f" | {lab}: all {ma['acc@1']:.3f}/{ma['ndcg@5']:.3f} -> seen {ms['acc@1']:.3f}"
                    f"+-{1.96 * se:.3f}/{ms['ndcg@5']:.3f} (z={z:+.1f}{flag})")
        sup = SUP.get(c, (float("nan"),) * 2)
        logger.info(msg + f" || STHGCN {sup[0]:.3f}/{sup[1]:.3f}")
        rows.append(r)
    logger.info("=== macro Acc@1 over cities (all -> STHGCN-style subset) ===")
    for lab, _ in specs:
        al = [r[lab]["all"]["acc@1"] for r in rows if lab in r]
        se = [r[lab]["seen"]["acc@1"] for r in rows if lab in r]
        sup = [SUP[r["city"]][0] for r in rows if lab in r and r["city"] in SUP]
        if al:
            logger.info(f"{lab:>14}: all {np.mean(al):.4f} | seen-subset {np.mean(se):.4f} | "
                        f"STHGCN {np.mean(sup):.4f} | cities above STHGCN (subset): "
                        f"{sum(r[lab]['seen']['acc@1'] > SUP[r['city']][0] for r in rows if lab in r and r['city'] in SUP)}/{len(sup)}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
