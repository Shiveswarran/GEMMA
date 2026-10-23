"""Decision table for scripts/run_gonogo.sh.

Pools per-query ranks over the held-out cities (paired: same queries in both arms),
reports micro (pooled queries) and macro (mean over cities) Acc@1, a paired bootstrap
95% CI of the pooled difference, and how many cities agree in sign.  Then applies the
pre-registered decision rules D1-D4.

    python scripts/gonogo_report.py [--runs runs] [--boot 5000]
"""
import argparse
import os

import numpy as np

TGT = "tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung".split()
LV = "tokyo shanghai melbourne sydney moscow".split()


def load(runs, run, ret, shot, tag, cities):
    out = {}
    for c in cities:
        f = os.path.join(runs, run, f"ranks_{ret}_{c}_{shot}{tag}.npy")
        if os.path.exists(f):
            out[c] = np.load(f)
    return out


def acc1(r):
    return (r < 1).astype(float)


def compare(name, A, B, boot, rng):
    """A, B: dict city -> 0-based ranks (paired). Returns pooled diff or None."""
    cs = [c for c in A if c in B and len(A[c]) == len(B[c])]
    if not cs:
        print(f"  {name:58s} (missing)")
        return None
    a = np.concatenate([acc1(A[c]) for c in cs]); b = np.concatenate([acc1(B[c]) for c in cs])
    d = a - b
    idx = rng.randint(0, len(d), size=(boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    macro = np.mean([acc1(A[c]).mean() - acc1(B[c]).mean() for c in cs])
    pos = sum(acc1(A[c]).mean() > acc1(B[c]).mean() for c in cs)
    sig = "*" if (lo > 0 or hi < 0) else " "
    print(f"  {name:58s} A={a.mean():.4f} B={b.mean():.4f} d={d.mean():+.4f}{sig} "
          f"CI[{lo:+.4f},{hi:+.4f}] macro-d={macro:+.4f} cities+={pos}/{len(cs)} n={len(d)}")
    return {"d": d.mean(), "lo": lo, "hi": hi, "A": a.mean(), "B": b.mean()}


def level(name, A):
    if not A:
        print(f"  {name:58s} (missing)"); return None
    a = np.concatenate([acc1(v) for v in A.values()])
    macro = np.mean([acc1(v).mean() for v in A.values()])
    print(f"  {name:58s} micro Acc@1={a.mean():.4f} macro={macro:.4f} cities={len(A)} n={len(a)}")
    return a.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs"); p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--tgt", default=" ".join(TGT)); p.add_argument("--lv", default=" ".join(LV))
    x = p.parse_args(); rng = np.random.RandomState(0)
    TGT[:] = x.tgt.split(); LV[:] = x.lv.split()
    L = lambda run, ret, shot, tag, cities=None: load(x.runs, run, ret, shot, tag, cities or TGT)
    V = {}

    print("\nG0  zero-shot, held-out cities: " + " ".join(TGT))
    hand, v6, noj = L("abl_incity", "hrem", "zero_shot", "_G0"), L("gemma_v4", "hrem", "zero_shot", "_G0"), \
        L("abl_random", "hrem", "zero_shot", "_G0")
    for n, a in (("hand evidence (no learning)", hand), ("v6 HREM (JEPA-v4 + source)", v6),
                 ("HREM, no JEPA (random init + source)", noj)):
        level(n, a)
    V["src_vs_hand"] = compare("source training vs hand evidence", v6, hand, x.boot, rng)
    V["jepa_z"] = compare("JEPA-v4 init vs random init (zero-shot)", v6, noj, x.boot, rng)

    print("\nG1  value of target labels (cities: %s)" % " ".join(LV))
    tz = {c: v for c, v in v6.items() if c in LV}
    tf, tF = L("gemma_v4", "hrem", "few_shot", "_G1", LV), L("gemma_v4", "hrem", "full", "_G1", LV)
    i5, iF = L("abl_incity", "hrem", "few_shot", "_G1", LV), L("abl_incity", "hrem", "full", "_G1", LV)
    V["label_value_incity"] = compare("in-city full  vs transfer zero-shot", iF, tz, x.boot, rng)
    compare("transfer full vs transfer zero-shot", tF, tz, x.boot, rng)
    V["transfer_5"] = compare("transfer 5%   vs in-city 5%", tf, i5, x.boot, rng)
    V["transfer_full"] = compare("transfer full vs in-city full", tF, iF, x.boot, rng)

    print("\nG2  cold start (history budget k past trajectories), zero-shot")
    V["cold"] = {}
    for k in (0, 1, 3):
        h = L("abl_incity", "hrem", "zero_shot", f"_G2h{k}"); v = L("gemma_v4", "hrem", "zero_shot", f"_G2h{k}")
        r = L("abl_random", "hrem", "zero_shot", f"_G2h{k}")
        level(f"k={k}: v6 HREM", v)
        V["cold"][k] = compare(f"k={k}: JEPA init vs random init", v, r, x.boot, rng)
        compare(f"k={k}: source-trained vs hand evidence", v, h, x.boot, rng)

    print("\nG3  source scaling (zero-shot, with evidence)")
    s = {n: L(f"scale_{n}", "hrem", "zero_shot", "_G3") for n in (1, 3, 6)}
    for n in (1, 3, 6):
        level(f"{n} source cities", s[n])
    V["scale_3"] = compare("3 vs 1 source cities", s[3], s[1], x.boot, rng)
    V["scale_6"] = compare("6 vs 1 source cities", s[6], s[1], x.boot, rng)

    print("\nG4  backbone only (NO hand evidence), zero-shot")
    bj, br = L("bo_jepa", "hrem", "zero_shot", "_G4"), L("bo_rand", "hrem", "zero_shot", "_G4")
    bg, b1 = L("bo_gen", "wm", "zero_shot", "_G4"), L("bo_jepa1", "hrem", "zero_shot", "_G4")
    for n, a in (("JEPA init + source", bj), ("random init + source", br),
                 ("GEMMA-G generative (random init)", bg), ("JEPA init + 1 source city", b1)):
        level(n, a)
    V["bo_jepa"] = compare("JEPA init vs random init", bj, br, x.boot, rng)
    V["bo_gen"] = compare("generative (GEMMA-G) vs random-init supervised", bg, br, x.boot, rng)
    V["bo_scale"] = compare("6 vs 1 source cities (backbone only)", bj, b1, x.boot, rng)
    best = max((a for a in (bj, br, bg) if a), key=lambda a: np.concatenate([acc1(v) for v in a.values()]).mean(),
               default=None)
    V["bo_gap"] = compare("best backbone-only vs v6 HREM (with evidence)", best or {}, v6, x.boot, rng)

    # ------------------------------------------------------------- decisions
    def g(k, f="d"):
        v = V.get(k); return None if v is None else v[f]
    print("\n================ DECISIONS (pre-registered) ================")
    lv = g("label_value_incity")
    if lv is not None:
        print(f"D1 label value = {lv:+.3f}: " + (
            "SATURATED -> full-vocab Acc@1 cannot show a big pretraining gain; centre the paper on "
            "zero/few-shot, cold start and new places" if lv < 0.02 else
            "labels still matter -> pretraining has room to substitute for them"))
    c0 = V["cold"].get(0) or V["cold"].get(1)
    if c0 is not None:
        print(f"D2 pretraining gain at cold start = {c0['d']:+.3f} (CI {c0['lo']:+.3f},{c0['hi']:+.3f}): " + (
            "YES -> pretraining carries transferable knowledge; cold start is the headline regime"
            if c0["lo"] > 0 and c0["d"] >= 0.01 else "NO detectable transferable knowledge in the backbone"))
    s6, bs = g("scale_6"), g("bo_scale")
    if s6 is not None or bs is not None:
        up = any(v is not None and V[k]["lo"] > 0 for k, v in (("scale_6", s6), ("bo_scale", bs)))
        print(f"D3 scaling 1->6 cities: with evidence {s6 if s6 is None else round(s6, 4)}, backbone-only "
              f"{bs if bs is None else round(bs, 4)}: " + (
                  "RISING -> build the STD-scale corpus (~30M check-ins) and pretrain on it" if up else
                  "FLAT -> more of the same data will not help; do not spend months on scale"))
    gap = g("bo_gap")
    if gap is not None:
        print(f"D4 best backbone-only minus v6 HREM = {gap:+.3f}: " + (
            "backbone is competitive -> 'strong pretrained backbone' claim is viable"
            if gap > -0.02 else "backbone is NOT the carrier -> present GEMMA as a structured "
            "return-explore model with a pretrained encoder, not as a backbone-first foundation model"))
    gg = g("bo_gen")
    if gg is not None:
        print(f"    generative (GEMMA-G) vs supervised, backbone only: {gg:+.3f} -> "
              + ("adopt GEMMA-G as the pretraining objective" if V['bo_gen']['lo'] > 0 else
                 "no evidence that the objective matters at this data scale"))


if __name__ == "__main__":
    main()
