"""Exploration law per city: P(new place | S places visited so far) ~ S^-gamma.

python scripts/mobility_laws.py --data_root data --cities istanbul ... bandung
Prints, per city, the return share of check-ins and the fitted exponent gamma (log-log least
squares over S with at least 30 observations), to compare with the learned gate coefficient
-g_S of a source-trained model (printed with --source).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def city_gamma(df, max_s=200, min_obs=30):
    num, den = np.zeros(max_s + 1), np.zeros(max_s + 1)
    ret = tot = 0
    for seq in df.sort_values("timestamp").groupby("user_id")["venue_id"].apply(list):
        seen = set()
        for i, v in enumerate(seq):
            S = len(seen)
            if i > 0:
                tot += 1; ret += v in seen
                if S <= max_s:
                    den[S] += 1; num[S] += v not in seen
            seen.add(v)
    S = np.arange(max_s + 1)
    ok = (den >= min_obs) & (S >= 1) & (num > 0)
    gamma = -np.polyfit(np.log(S[ok]), np.log(num[ok] / den[ok]), 1)[0] if ok.sum() >= 3 else np.nan
    return ret / max(tot, 1), gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--source", default=None, help="source-trained model to print -g_S")
    a = ap.parse_args()
    gammas = []
    print(f"{'city':>14} {'return share':>12} {'gamma':>7}")
    for c in a.cities:
        files = glob.glob(os.path.join(a.data_root, c, f"{c}_checkins_*.csv"))
        df = pd.concat([pd.read_csv(f, usecols=["user_id", "venue_id", "timestamp"]) for f in files])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        rs, g = city_gamma(df.dropna())
        gammas.append(g)
        print(f"{c:>14} {rs:12.3f} {g:7.3f}")
    print(f"gamma: median {np.nanmedian(gammas):.3f}, IQR [{np.nanpercentile(gammas, 25):.3f}, "
          f"{np.nanpercentile(gammas, 75):.3f}]")
    if a.source:
        import torch
        ck = torch.load(a.source, map_location="cpu", weights_only=False)
        print(f"learned gate: -g_S = {-float(ck['head']['g_S']):.3f}")


if __name__ == "__main__":
    main()
