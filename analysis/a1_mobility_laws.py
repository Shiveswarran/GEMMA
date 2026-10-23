"""A1 — Universal mobility laws across 15 cities (problem statement / motivation).

Claim supported: the statistics that decide the next place are nearly the SAME in every city,
even though no POI is shared between cities -> a model of these laws transfers zero-shot.

Per city (all splits, per-user time order):
  * revisit share: fraction of check-ins at a place the user visited before (preferential return)
  * unseen-target share of test targets (what the STHGCN protocol removes)
  * exploration law  S(n) ~ n^mu   (distinct places after n check-ins; EPR, Song et al. 2010)
  * novelty law      P(new | S) ~ S^-gamma
  * visitation Zipf  f_k ~ k^-alpha  (k-th most visited place of a user)
Outputs: table A1_mobility_laws, figures A1a_exploration_law, A1b_zipf, A1c_novelty, A1d_revisit_share
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import (SRC, TGT, base_args, load_cfg, taxonomy, load_city, save_table, setup_mpl,
                    save_fig, pretty, city_style, sthgcn_test_queries, dump_json)


def fit_loglog(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1, w=None if w is None else np.asarray(w)[ok])[0])


def city_laws(C, max_n=200, max_k=20):
    df = C["df"]
    seq = df.groupby("user_id")["venue_id"].apply(list)
    n_new_at = np.zeros(max_n + 1); n_at = np.zeros(max_n + 1)
    S_sum = np.zeros(max_n + 1); S_cnt = np.zeros(max_n + 1)
    rev, tot = 0, 0
    pnew_S_num = np.zeros(max_n + 1); pnew_S_den = np.zeros(max_n + 1)
    zipf = np.zeros(max_k); zipf_users = 0
    for v in seq:
        seen = set()
        for i, p in enumerate(v):
            S = len(seen)
            if i > 0:
                tot += 1; rev += p in seen
                if S <= max_n:
                    pnew_S_den[S] += 1; pnew_S_num[S] += p not in seen
            seen.add(p)
            n = i + 1
            if n <= max_n:
                S_sum[n] += len(seen); S_cnt[n] += 1
        cnt = pd.Series(v).value_counts().values
        if len(cnt) >= 5:
            f = cnt[:max_k] / cnt.sum()
            zipf[:len(f)] += f; zipf_users += 1
    n = np.arange(max_n + 1)
    Sn = np.where(S_cnt > 0, S_sum / np.maximum(S_cnt, 1), np.nan)
    pnew = np.where(pnew_S_den >= 30, pnew_S_num / np.maximum(pnew_S_den, 1), np.nan)
    zipf = zipf / max(zipf_users, 1)
    sel = (n >= 2) & (S_cnt >= 30)
    return {"revisit_share": rev / max(tot, 1), "Sn": Sn, "pnew": pnew, "zipf": zipf,
            "mu": fit_loglog(n[sel], Sn[sel]), "gamma": -fit_loglog(n[1:], pnew[1:]),
            "alpha": -fit_loglog(np.arange(1, max_k + 1), zipf), "zipf_users": zipf_users}


def main():
    a = base_args(__doc__).parse_args()
    cfg = load_cfg(a); fsq, _ = taxonomy(cfg)
    plt = setup_mpl()
    rows, curves = [], {}
    for c in a.cities:
        try:
            C = load_city(cfg, fsq, c)
        except FileNotFoundError:
            print(f"skip {c}: no data"); continue
        df = C["df"]
        L = city_laws(C)
        q, _ = sthgcn_test_queries(C)
        n_test = sum(1 for t in C["trails"] if t["split"] == "test" and len(t["rows"]) >= 2)
        rows.append({"city": pretty(c), "role": "source" if c in SRC else "target",
                     "users": df["user_id"].nunique(), "check-ins": len(df), "POIs": df["venue_id"].nunique(),
                     "traj_len": float(np.mean([len(t["rows"]) for t in C["trails"]])),
                     "revisit_share": L["revisit_share"], "unseen_test_target": 1 - len(q) / max(n_test, 1),
                     "mu (S~n^mu)": L["mu"], "gamma (Pnew~S^-g)": L["gamma"], "alpha (Zipf)": L["alpha"]})
        curves[c] = L
    T = pd.DataFrame(rows)
    save_table(T, a.out, "A1_mobility_laws",
               "Mobility statistics of the 15 Massive-STEPS cities. The exponents of the exploration law "
               "($\\mu$), novelty law ($\\gamma$) and visitation Zipf law ($\\alpha$) are close across "
               "cities although no POI is shared between them.")
    dump_json({c: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in L.items()} for c, L in curves.items()},
              a.out, "A1_curves")
    if not curves:
        return
    cmap = plt.get_cmap("tab20")
    for name, key, xl, yl, xs in (("A1a_exploration_law", "Sn", "check-ins so far $n$", "distinct places $S(n)$", None),
                                  ("A1c_novelty_law", "pnew", "distinct places so far $S$", "$P$(next place is new)", None),
                                  ("A1b_visitation_zipf", "zipf", "rank $k$ of place in user's history", "visit share $f_k$", None)):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for i, (c, L) in enumerate(curves.items()):
            y = L[key]; x = np.arange(1, len(y) + 1) if key == "zipf" else np.arange(len(y))
            ok = np.isfinite(y) & (y > 0) & (x > 0)
            ax.plot(x[ok], y[ok], color=cmap(i % 20), label=pretty(c), **city_style(c))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(ncol=2, frameon=False, fontsize=6)
        save_fig(fig, a.out, name); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    T2 = T.sort_values("revisit_share")
    col = ["#7F7F7F" if r == "source" else "#0072B2" for r in T2["role"]]
    ax.barh(T2["city"], T2["revisit_share"], color=col)
    ax.barh(T2["city"], T2["unseen_test_target"], left=0, color="none", edgecolor="#D55E00", hatch="///", lw=0.6)
    ax.set_xlabel("share of check-ins that revisit a known place (bars) /\nunseen test targets (hatched)")
    save_fig(fig, a.out, "A1d_revisit_share"); plt.close(fig)


if __name__ == "__main__":
    main()
