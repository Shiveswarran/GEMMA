"""Paper tables from saved per-query ranks (all runs already use the STHGCN test protocol).

python scripts/final_report.py --baseline v6 \
   --runs "v6=runs/gemma_v4:hrem:_v6" "hand=runs/abl_incity:hrem:_hand" \
          "GEMMA=runs/gemma_final_s0:wm:_final" \
   --cities tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung \
   --shot zero_shot

For each city: Acc@1 / NDCG@5 of every run, z-test against the published STHGCN Acc@1
(treated as a constant), paired bootstrap of every run against --baseline, and a macro row.
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gemma.sota_ref import SUP


def m(rk):
    return {"acc@1": float((rk < 1).mean()),
            "ndcg@5": float(np.where(rk < 5, 1.0 / np.log2(rk + 2), 0.0).mean()),
            "mrr": float((1.0 / (rk + 1)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir:retriever:tag")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--shot", default="zero_shot")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    runs = []
    for s in a.runs:
        lab, rest = s.split("=", 1)
        rd, ret, tag = (rest.split(":") + ["", ""])[:3]
        runs.append((lab, rd, ret, tag))
    rng = np.random.RandomState(0)
    macro = {lab: [] for lab, *_ in runs}; wins = {lab: 0 for lab, *_ in runs}
    sigw = {lab: 0 for lab, *_ in runs}; sigl = {lab: 0 for lab, *_ in runs}
    hdr = f"{'city':>13} {'n':>6} {'STHGCN':>7} | " + " | ".join(f"{lab:>22}" for lab, *_ in runs)
    print(hdr); print("-" * len(hdr))
    for c in a.cities:
        rks = {}
        for lab, rd, ret, tag in runs:
            f = os.path.join(rd, f"ranks_{ret}_{c}_{a.shot}{tag}.npy")
            if os.path.exists(f):
                rks[lab] = np.load(f)
        if not rks:
            print(f"{c:>13}  (no ranks)"); continue
        n = len(next(iter(rks.values())))
        ref = SUP.get(c, (float("nan"),))[0]
        cells = []
        for lab, *_ in runs:
            if lab not in rks or len(rks[lab]) != n:
                cells.append(f"{'missing' if lab not in rks else 'len!=':>22}"); continue
            mm = m(rks[lab]); p = mm["acc@1"]
            z = (p - ref) / math.sqrt(max(p * (1 - p), 1e-9) / n)
            mark = ">" if z > 1.96 else ("<" if z < -1.96 else "=")
            sigw[lab] += z > 1.96; sigl[lab] += z < -1.96; wins[lab] += p > ref
            extra = ""
            if a.baseline and lab != a.baseline and a.baseline in rks and len(rks[a.baseline]) == n:
                d = (rks[lab] < 1).astype(float) - (rks[a.baseline] < 1).astype(float)
                bs = d[rng.randint(0, n, size=(a.boot, n))].mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                extra = "*" if (lo > 0 or hi < 0) else ""
                extra = f" d{d.mean():+.3f}{extra}"
            macro[lab].append((p, mm["ndcg@5"], ref))
            cells.append(f"{p:.3f}/{mm['ndcg@5']:.3f}{mark}{extra:>8}")
        print(f"{c:>13} {n:>6} {ref:>7.3f} | " + " | ".join(cells))
    print("-" * len(hdr))
    for lab, *_ in runs:
        if macro[lab]:
            acc = np.mean([x[0] for x in macro[lab]]); nd = np.mean([x[1] for x in macro[lab]])
            ref = np.mean([x[2] for x in macro[lab]])
            print(f"{lab:>10}: macro Acc@1 {acc:.4f} (STHGCN {ref:.4f}, {acc - ref:+.4f}) NDCG@5 {nd:.4f} | "
                  f"above STHGCN {wins[lab]}/{len(macro[lab])}, significantly above {sigw[lab]}, "
                  f"significantly below {sigl[lab]}")
    print("legend: '>' / '<' = z-test vs published STHGCN Acc@1 at 95%;  d = paired Acc@1 difference vs "
          f"baseline '{a.baseline}', '*' = 95% bootstrap CI excludes 0")


if __name__ == "__main__":
    main()
