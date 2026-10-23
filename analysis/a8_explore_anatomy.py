"""A8 — Anatomy of exploration: why is Acc@1 on never-visited places ~0, and what information would fix it?

Explore query = an STHGCN-protocol test target that the user has never visited before (not in the user's
train trajectories nor in the current trip).  Candidates = every POI of the city's train split that the
user has not visited.  No model is needed; every ranker below is a closed-form oracle or prior.

Questions answered
  (1) How hard is exploration?  Acc@1/@10 and mean log2-rank of simple rankers, with ORACLE knowledge
      of the target's activity or leaf category (an upper bound on what an activity/category predictor buys).
  (2) Bits budget: log2|candidates| - E[log2 rank] = information each signal supplies.  Which bits are
      available zero-shot (geometry, activity) and which need target-city logs (popularity)?
  (3) Is there a city-invariant law of exploration?  Opportunity rank (Noulas et al. 2012): rank of the
      chosen place among all places ordered by distance from the current place; P(rank) ~ rank^-alpha.
      If alpha is the same in every city, an explore prior in RANK space transfers zero-shot while a
      prior in kilometres does not.
  (4) Is exploration bursty?  P(new | previous check-in was new) vs P(new | previous was a return):
      a free, transferable feature for the return/explore gate.

Outputs: tables A8_explore_oracles, A8_bits, A8_rank_law, A8_momentum; figures A8a_rank_law,
A8b_oracle_ladder.   CPU only.
  python analysis/a8_explore_anatomy.py --run_dir runs/gemma_v4 --data_root data --cities tokyo ... bandung
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import (SRC, TGT, base_args, load_cfg, taxonomy, load_city, sthgcn_test_queries, user_history,
                    save_table, setup_mpl, save_fig, pretty, city_style)


def haversine_km(lat0, lon0, lat, lon):
    p0, p = np.radians(lat0), np.radians(lat)
    h = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def poi_table(df, train_rows):
    tr = df.loc[train_rows]
    g = df.groupby("venue_id")
    P = pd.DataFrame({"lat": g["lat"].mean(), "lon": g["lon"].mean(),
                      "act": g["act"].agg(lambda x: x.mode().iat[0]),
                      "cat": g["venue_category_id"].agg(lambda x: x.mode().iat[0])})
    P["pop"] = tr.groupby("venue_id")["user_id"].nunique().reindex(P.index).fillna(0).values
    P = P[P.index.isin(set(tr["venue_id"]))].dropna(subset=["lat", "lon"])      # STHGCN vocabulary
    return P


def rank_of(score, t, higher=True):
    """1-based rank of index t, ties broken pessimistically-averaged (expected rank under random tie order)."""
    s = score[t]
    better = (score > s).sum() if higher else (score < s).sum()
    ties = (score == s).sum() - 1
    return better + 1 + ties / 2.0


def analyse_city(C, max_q, rng):
    df = C["df"]
    Q, train = sthgcn_test_queries(C)
    hist = user_history(train)
    train_rows = np.concatenate([t["rows"] for t in train]) if train else np.array([], int)
    P = poi_table(df, train_rows)
    vid = P.index.to_numpy(); pos = {v: i for i, v in enumerate(vid)}
    lat, lon, act, cat, pop = (P["lat"].to_numpy(), P["lon"].to_numpy(), P["act"].to_numpy(),
                               P["cat"].astype(str).to_numpy(), P["pop"].to_numpy().astype(float))
    n_rev = 0; recs = []
    order = rng.permutation(len(Q))
    for qi in order:
        t, r = Q[qi]
        rows = list(t["rows"]); j = rows.index(r)
        if j < 1:
            continue
        H = set(df.loc[np.concatenate([h["rows"] for h in hist.get(t["user_id"], [])] + [np.array(rows[:j])]),
                       "venue_id"].tolist())
        v = df.at[r, "venue_id"]
        if v in H:
            n_rev += 1; continue
        cur = df.at[rows[j - 1], "venue_id"]
        if v not in pos or cur not in pos:
            continue
        ti = pos[v]
        cand = np.ones(len(vid), bool)
        for h in H:
            if h in pos:
                cand[pos[h]] = False
        idx = np.where(cand)[0]; tt = int(np.where(idx == ti)[0][0])
        d = haversine_km(lat[pos[cur]], lon[pos[cur]], lat[idx], lon[idx])
        dall = haversine_km(lat[pos[cur]], lon[pos[cur]], lat, lon)
        same_a = act[idx] == act[ti]; same_c = cat[idx] == cat[ti]
        rec = {"n_cand": len(idx)}
        rec["nearest"] = rank_of(-d, tt)
        ia = np.where(same_a)[0]; rec["nearest | activity*"] = rank_of(-d[ia], int(np.where(ia == tt)[0][0]))
        ic = np.where(same_c)[0]; rec["nearest | category*"] = rank_of(-d[ic], int(np.where(ic == tt)[0][0]))
        rec["popularity"] = rank_of(pop[idx], tt)
        rec["popularity | category*"] = rank_of(pop[idx][ic], int(np.where(ic == tt)[0][0]))
        grav = np.log1p(pop[idx]) - np.log(d + 0.1)
        rec["gravity (pop/dist)"] = rank_of(grav, tt)
        rec["gravity | activity*"] = rank_of(grav[ia], int(np.where(ia == tt)[0][0]))
        # opportunity rank (Noulas et al.): places (ALL, visited or not) closer than the chosen one, +1
        rec["opp_rank_all"] = int((dall < dall[ti]).sum()) + 1
        rec["opp_rank_act"] = int((dall[act == act[ti]] < dall[ti]).sum()) + 1
        rec["dist_km"] = float(dall[ti])
        recs.append(rec)
        if len(recs) >= max_q:
            break
    return pd.DataFrame(recs), n_rev


def momentum(C):
    df = C["df"]; tr = [t for t in C["trails"] if t["split"] == "train"]
    by_u = {}
    for t in sorted(tr, key=lambda t: t["t0"]):
        by_u.setdefault(t["user_id"], []).append(t)
    nn = nr = rn = rr = 0
    for u, ts in by_u.items():
        seen = set(); prev_new = None
        for t in ts:
            for r in t["rows"]:
                v = df.at[r, "venue_id"]; new = v not in seen
                if prev_new is not None:
                    if prev_new: nn += new; nr += 1
                    else: rn += new; rr += 1
                if seen:
                    prev_new = new
                seen.add(v)
    return nn / max(nr, 1), rn / max(rr, 1), nr + rr


def fit_alpha(ranks, rmax=2000):
    """Discrete power-law exponent by log-binned least squares on P(rank) for 1 <= rank <= rmax."""
    r = np.asarray(ranks); r = r[(r >= 1) & (r <= rmax)]
    if len(r) < 30:
        return float("nan"), None
    edges = np.unique(np.round(np.logspace(0, np.log10(rmax), 25)).astype(int))
    cnt, _ = np.histogram(r, bins=edges)
    dens = cnt / np.diff(edges) / len(r)
    mid = np.sqrt(edges[:-1] * edges[1:])
    ok = dens > 0
    if ok.sum() < 4:
        return float("nan"), None
    b = np.polyfit(np.log(mid[ok]), np.log(dens[ok]), 1)
    return float(-b[0]), (mid[ok], dens[ok])


def main():
    ap = base_args(__doc__, TGT)
    ap.add_argument("--max_queries", type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    cfg = load_cfg(a); fsq, names = taxonomy(cfg)
    plt = setup_mpl()
    RANKERS = ["nearest", "nearest | activity*", "nearest | category*", "popularity", "popularity | category*",
               "gravity (pop/dist)", "gravity | activity*"]
    rows, bits, law, mom, curves = [], [], [], [], {}
    for c in a.cities:
        try:
            C = load_city(cfg, fsq, c)
        except FileNotFoundError:
            print(f"skip {c}"); continue
        R, n_rev = analyse_city(C, a.max_queries, rng)
        if R.empty:
            print(f"[{c}] no explore queries"); continue
        n = len(R)
        print(f"[{c}] explore queries={n} (revisits skipped={n_rev}) median candidates={int(R['n_cand'].median())}")
        row = {"city": pretty(c), "explore n": n, "explore share": n / max(n + n_rev, 1),
               "median candidates": int(R["n_cand"].median())}
        b = {"city": pretty(c), "log2 |candidates|": float(np.log2(R["n_cand"]).mean())}
        for k in RANKERS:
            row[f"{k} @1"] = float((R[k] <= 1).mean()); row[f"{k} @10"] = float((R[k] <= 10).mean())
            b[k] = float(np.log2(R["n_cand"]).mean() - np.log2(R[k]).mean())        # bits gained
        rows.append(row); bits.append(b)
        aa, ca = fit_alpha(R["opp_rank_all"]); ab, cb = fit_alpha(R["opp_rank_act"])
        curves[c] = (ca, cb, R["dist_km"].to_numpy())
        law.append({"city": pretty(c), "alpha (all places)": aa, "alpha (same activity)": ab,
                    "median km": float(R["dist_km"].median()), "median opp. rank (all)": float(R["opp_rank_all"].median())})
        pn, pr_, m = momentum(C)
        mom.append({"city": pretty(c), "P(new | prev new)": pn, "P(new | prev return)": pr_,
                    "ratio": pn / max(pr_, 1e-9), "transitions": m})
    if not rows:
        return
    def with_mean(L):
        T = pd.DataFrame(L)
        m = {"city": "mean"}; m.update({k: float(T[k].astype(float).mean()) for k in T.columns if k != "city"})
        return pd.concat([T, pd.DataFrame([m])], ignore_index=True)
    O = with_mean(rows)
    save_table(O, a.out, "A8_explore_oracles",
               "Exploration (never-visited targets, STHGCN protocol): Acc@1/@10 of closed-form rankers over the "
               "user's unvisited POIs.  * = oracle knowledge of the target's activity / leaf category.  "
               "Popularity and gravity use the target city's train log (not available strictly zero-shot).")
    save_table(with_mean(bits), a.out, "A8_bits",
               "Information supplied by each signal for exploration: log2|candidates| - E[log2 rank] (bits).")
    save_table(with_mean(law), a.out, "A8_rank_law",
               "Opportunity-rank law of exploration (Noulas et al. 2012): P(rank) ~ rank^-alpha, rank = number of "
               "places closer to the current place than the chosen one, +1.  Similar alpha across cities = a "
               "city-invariant exploration prior.")
    save_table(with_mean(mom), a.out, "A8_momentum",
               "Exploration is bursty: probability that a check-in is a first visit given the previous one was / was not.")
    # figures: rank-law collapse vs. distance
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    cmap = plt.get_cmap("tab20")
    for i, (c, (ca, cb, dk)) in enumerate(curves.items()):
        if ca is not None:
            axes[0].loglog(ca[0], ca[1], "o-", ms=2, color=cmap(i % 20), label=pretty(c), **city_style(c))
        h, e = np.histogram(dk[dk > 0], bins=np.logspace(-2, 2, 25), density=True)
        mid = np.sqrt(e[:-1] * e[1:]); ok = h > 0
        axes[1].loglog(mid[ok], h[ok], "o-", ms=2, color=cmap(i % 20), **city_style(c))
    xs = np.logspace(0, 3, 10); axes[0].loglog(xs, 0.5 * xs ** -0.84, "k--", lw=0.8, label="rank$^{-0.84}$")
    axes[0].set_xlabel("opportunity rank of the chosen new place"); axes[0].set_ylabel("P(rank)")
    axes[1].set_xlabel("distance to the chosen new place (km)"); axes[1].set_ylabel("density")
    axes[0].legend(fontsize=5, frameon=False, ncol=2)
    save_fig(fig, a.out, "A8a_rank_law"); plt.close(fig)
    m = O.iloc[-1]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    x = np.arange(len(RANKERS))
    ax.bar(x - 0.2, [m[f"{k} @1"] for k in RANKERS], 0.4, label="Acc@1", color="#0072B2")
    ax.bar(x + 0.2, [m[f"{k} @10"] for k in RANKERS], 0.4, label="Acc@10", color="#56B4E9")
    ax.set_xticks(x); ax.set_xticklabels(RANKERS, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("explore accuracy (macro)"); ax.legend(frameon=False)
    save_fig(fig, a.out, "A8b_oracle_ladder"); plt.close(fig)


if __name__ == "__main__":
    main()
