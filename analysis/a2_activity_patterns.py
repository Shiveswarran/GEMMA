"""A2 — Activity is the transferable unit: the abstraction ladder (POI -> category -> activity).

Claim supported: cities share NO places, few category frequencies, but nearly identical
activity rhythms (activity x hour) and activity transitions.  The more abstract the
representation, the more similar cities become -> GEMMA tokenises check-ins by activity /
category / time / relative space instead of POI ids, and that is what transfers.

Outputs
  table  A2_abstraction_ladder   mean pairwise cross-city Jensen-Shannon divergence (bits) per level
  table  A2_revisit_by_activity  P(revisit | activity of target) per city (motivates activity-gated return/explore)
  figure A2a_ladder              bar chart of the ladder
  figure A2b_js_activity_hour    15x15 city-similarity heatmap (activity x hour)
  figure A2c_rhythms             P(activity | hour) for 4 activities, every city overlaid
  figure A2d_revisit_by_activity heatmap city x activity
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import (SRC, TGT, base_args, load_cfg, taxonomy, load_city, save_table, setup_mpl, save_fig,
                    pretty, city_style, js_div, short_act)


def dists(C, A, cat_index, pri_index):
    df = C["df"]
    acts = df["act"].values; hours = df["hour"].values
    act_hour = np.zeros((A, 24)); np.add.at(act_hour, (acts, hours), 1)
    act = act_hour.sum(1)
    cat = np.zeros(len(cat_index)); pri = np.zeros(len(pri_index))
    for x in df["venue_category_id"].astype(str):
        if x in cat_index: cat[cat_index[x]] += 1
    for x in df["pri"]:
        pri[pri_index[x]] += 1
    trans = np.zeros((A, A))
    for t in C["trails"]:
        a_ = acts[t["rows"]]
        np.add.at(trans, (a_[:-1], a_[1:]), 1)
    # revisit by activity of the target (user-level, time order)
    rev = np.zeros(A); tot = np.zeros(A)
    for _, g in df.groupby("user_id"):
        seen = set()
        for v, a_ in zip(g["venue_id"].values, g["act"].values):
            if seen:
                tot[a_] += 1; rev[a_] += v in seen
            seen.add(v)
    return {"act": act, "act_hour": act_hour, "cat": cat, "pri": pri, "trans": trans,
            "pois": df["venue_id"].value_counts().to_dict(), "rev": rev, "rev_tot": tot}


def main():
    a = base_args(__doc__).parse_args()
    cfg = load_cfg(a); fsq, names = taxonomy(cfg)
    A = max(fsq.act2id.values()) + 1
    plt = setup_mpl()
    cats = sorted(fsq.id2levels.keys()); cat_index = {c: i for i, c in enumerate(cats)}
    pri_index = dict(fsq.pri2id)
    D = {}
    for c in a.cities:
        try:
            C = load_city(cfg, fsq, c)
        except FileNotFoundError:
            print(f"skip {c}"); continue
        C["df"]["pri"] = [fsq.primary_id(x) for x in C["df"]["venue_category_id"]]
        D[c] = dists(C, A, cat_index, {v: v for v in range(len(pri_index) + 1)})
    cities = list(D)
    if len(cities) < 2:
        print("need >= 2 cities"); return
    levels = {"POI (venue id)": None, "category (leaf)": "cat", "primary category": "pri",
              "activity": "act", "activity x hour": "act_hour", "activity transition": "trans"}
    rows = []
    J = {}
    for lab, key in levels.items():
        vals = []
        M = np.zeros((len(cities), len(cities)))
        for i, ci in enumerate(cities):
            for j, cj in enumerate(cities):
                if i >= j:
                    continue
                if key is None:     # POI visit distributions over the union vocabulary
                    keys = list(set(D[ci]["pois"]) | set(D[cj]["pois"]))
                    d = js_div([D[ci]["pois"].get(k, 0) for k in keys], [D[cj]["pois"].get(k, 0) for k in keys])
                else:
                    d = js_div(D[ci][key].ravel(), D[cj][key].ravel())
                M[i, j] = M[j, i] = d; vals.append(d)
        J[lab] = M
        rows.append({"representation": lab, "mean cross-city JS (bits)": float(np.nanmean(vals)),
                     "max": float(np.nanmax(vals)), "shared POIs (any pair)":
                     int(sum(len(set(D[x]["pois"]) & set(D[y]["pois"])) for x in cities for y in cities if x < y)) if key is None else ""})
    T = pd.DataFrame(rows)
    save_table(T, a.out, "A2_abstraction_ladder",
               "Mean pairwise Jensen--Shannon divergence between cities (bits; 0 = identical, 1 = disjoint). "
               "Cities share no POIs, but their activity rhythms and activity transitions are nearly identical.")
    act_ids = [k for k in range(1, A) if sum(D[c]["act"][k] for c in cities) > 0]
    R = pd.DataFrame({"city": [pretty(c) for c in cities],
                      **{short_act(names.get(k, str(k))): [D[c]["rev"][k] / D[c]["rev_tot"][k] if D[c]["rev_tot"][k] >= 20
                                                     else float("nan") for c in cities] for k in act_ids}})
    save_table(R, a.out, "A2_revisit_by_activity",
               "Probability that the next check-in revisits a known place, by activity of that check-in.")
    # ---- figures
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.barh(T["representation"][::-1], T["mean cross-city JS (bits)"][::-1],
            color=["#0072B2" if "activity" in s else "#999999" for s in T["representation"][::-1]])
    ax.set_xlabel("mean cross-city Jensen–Shannon divergence (bits)")
    ax.set_xlim(0, 1)
    save_fig(fig, a.out, "A2a_ladder"); plt.close(fig)
    M = J["activity x hour"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=max(0.05, np.nanmax(M)))
    ax.set_xticks(range(len(cities))); ax.set_yticks(range(len(cities)))
    ax.set_xticklabels([pretty(c) for c in cities], rotation=90, fontsize=6)
    ax.set_yticklabels([pretty(c) for c in cities], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="JS divergence (bits)")
    ax.set_title("activity × hour distributions")
    save_fig(fig, a.out, "A2b_js_activity_hour"); plt.close(fig)
    top = sorted(act_ids, key=lambda k: -sum(D[c]["act"][k] for c in cities))[:4]
    fig, axes = plt.subplots(1, len(top), figsize=(2.0 * len(top), 1.9), sharey=False)
    cmap = plt.get_cmap("tab20")
    for ax, k in zip(np.atleast_1d(axes), top):
        for i, c in enumerate(cities):
            ah = D[c]["act_hour"]; p = ah[k] / np.maximum(ah.sum(0), 1)
            ax.plot(range(24), p, color=cmap(i % 20), label=pretty(c), **city_style(c))
        ax.set_title(short_act(names.get(k, str(k))), fontsize=8); ax.set_xlabel("hour")
        ax.set_xticks([0, 6, 12, 18, 23])
    np.atleast_1d(axes)[0].set_ylabel("P(activity | hour)")
    np.atleast_1d(axes)[-1].legend(fontsize=5, frameon=False, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, a.out, "A2c_rhythms"); plt.close(fig)
    V = R.drop(columns=["city"]).values.astype(float)
    fig, ax = plt.subplots(figsize=(0.45 * V.shape[1] + 1.5, 0.28 * V.shape[0] + 1.0))
    im = ax.imshow(V, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(R))); ax.set_yticklabels(R["city"], fontsize=7)
    ax.set_xticks(range(V.shape[1])); ax.set_xticklabels(R.columns[1:], rotation=60, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="P(revisit | activity)")
    save_fig(fig, a.out, "A2d_revisit_by_activity"); plt.close(fig)


if __name__ == "__main__":
    main()
