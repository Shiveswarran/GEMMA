"""Paired significance test between two runs on the SAME test queries.

    python scripts/paired_test.py runs/gemma_wm/ranks_wm_tokyo_zero_shot.npy \
                                  runs/abl_wm_noJEPA/ranks_wm_tokyo_zero_shot.npy

Reports Acc@1/Acc@5/MRR of both, the difference, a paired bootstrap 95% CI and
an exact McNemar test on Acc@1.  With ~1.1k Tokyo queries the unpaired SE of
Acc@1 is ~0.013, but the paired SE is usually 2-3x smaller -- this is the test
to use for "pretraining helps" / "the world model helps" claims.
"""
import argparse
import math

import numpy as np


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial test with p = 0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    x = p.parse_args()
    ra, rb = np.load(x.a), np.load(x.b)
    assert ra.shape == rb.shape, f"different query sets: {ra.shape} vs {rb.shape}"
    n = len(ra)
    # ranks are 0-based (0 = top-1), as written by gemma.realize.evaluate
    mets = {"acc@1": lambda r: (r < 1).astype(float), "acc@5": lambda r: (r < 5).astype(float),
            "mrr": lambda r: 1.0 / (r + 1.0)}
    rng = np.random.RandomState(x.seed)
    idx = rng.randint(0, n, size=(x.boot, n))
    print(f"n={n}\n  A={x.a}\n  B={x.b}")
    for name, f in mets.items():
        fa, fb = f(ra), f(rb); d = fa - fb
        bs = d[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{name:6s} A={fa.mean():.4f} B={fb.mean():.4f} diff={d.mean():+.4f} "
              f"paired-SE={d.std(ddof=1) / math.sqrt(n):.4f} 95%CI=[{lo:+.4f},{hi:+.4f}]")
    ha, hb = ra < 1, rb < 1
    b, c = int((ha & ~hb).sum()), int((~ha & hb).sum())
    print(f"McNemar acc@1: A-only={b} B-only={c} p={mcnemar_exact(b, c):.4g}")


if __name__ == "__main__":
    main()
