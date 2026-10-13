"""A4 — What information decides the next place?  (and which of it GEMMA uses)

On the STHGCN test protocol of each city, every signal is turned into a one-signal ranker and
scored with Acc@1 / Acc@5 / NDCG@5 (ties broken uniformly at random, computed exactly).
History = the user's TRAIN trajectories (as in our zero-shot setting); "city" signals use the
whole train log of the city (what a supervised model such as STHGCN learns from).

  signal                     source of knowledge      used by GEMMA zero-shot?
  user visit frequency       the user's own history   yes (return channel)
  user recency               the user's own history   yes
  user transition last->v    the user's own history   yes
  user time-of-day match     the user's own history   yes
  distance to current place  geometry                 yes (both channels)
  city popularity            other users, same city   no  (needs target-city data)
  city transition last->v    other users, same city   no  (needs target-city data)
  activity prior x distance  activity model + space   yes (activity / explore channel)
plus coverage "oracles": share of targets in the user's history, within 1 km, etc.
Outputs: table A4_information_value (+ per-city csv), figure A4a_information_value
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from common import (SRC, TGT, base_args, load_cfg, taxonomy, load_city, save_table, setup_mpl, save_fig,
                    pretty, sthgcn_test_queries, user_history)

USED = {"user frequency": "yes", "user recency": "yes", "user transition": "yes", "user time-of-day": "yes",
        "distance": "yes", "activity prior x distance": "yes", "city popularity": "no (target data)",
        "city transition": "no (target data)"}


def hit_probs(scores, target, k_list=(1, 5)):
    """exact expected Acc@k / NDCG@5 under uniform random tie-breaking; scores: dict poi->score,
    POIs not in dict have score 0 (or -inf for distance-like handled by caller)."""
    s_t = scores.get(target, 0.0)
    vals = np.fromiter(scores.values(), float) if scores else np.zeros(0)
    g = int((vals > s_t).sum())
    t = int((vals == s_t).sum()) if target in scores else None
    return g, t, s_t


def score_metrics(g, ties):
    out = {}
    for k in (1, 5):
        out[f"acc@{k}"] = float(np.clip((k - g) / ties, 0, 1))
    nd = 0.0
    for pos in range(g, min(g + ties, 5)):       # target uniformly among tied positions
        nd += 1.0 / np.log2(pos + 2) / ties
    out["ndcg@5"] = nd
    return out


def main():
    ap = base_args(__doc__, TGT)
    ap.add_argument("--max_queries", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    cfg = load_cfg(a); fsq, names = taxonomy(cfg)
    plt = setup_mpl()
    rows, per_city = [], []
    for c in a.cities:
        try:
            C = load_city(cfg, fsq, c)
        except FileNotFoundError:
            print(f"skip {c}"); continue
        df = C["df"]
        Q, train = sthgcn_test_queries(C)
        if len(Q) > a.max_queries:
            Q = [Q[i] for i in rng.choice(len(Q), a.max_queries, replace=False)]
        H = user_history(train)
        vid = df["venue_id"].values; lat = df["lat"].values; lon = df["lon"].values
        hour = df["hour"].values; ts = df["timestamp"].values.astype("datetime64[s]").astype(np.int64)
        act = df["act"].values
        # city-level statistics from the train log
        trows = np.concatenate([t["rows"] for t in train])
        pop = Counter(vid[trows].tolist())
        trans = defaultdict(Counter)
        for t in train:
            v = vid[t["rows"]]
            for x, y in zip(v[:-1], v[1:]):
                trans[x][y] += 1
        poi_xy = df.groupby("venue_id")[["lat", "lon"]].mean()
        P_ids = poi_xy.index.values; P_lat = poi_xy["lat"].values; P_lon = poi_xy["lon"].values
        poi_act = df.groupby("venue_id")["act"].agg(lambda s: s.value_counts().index[0])
        P_act = poi_act.reindex(P_ids).values
        acc = defaultdict(list); cover = defaultdict(list)
        for t, r in Q:
            u = t["user_id"]; rows_ = t["rows"]; last = rows_[-2]; tgt = vid[r]
            hrows = np.concatenate([h["rows"] for h in H[u]]) if H[u] else np.zeros(0, int)
            hv = vid[hrows]
            freq = Counter(hv.tolist())
            rec = {}
            for rr in hrows:
                rec[vid[rr]] = max(rec.get(vid[rr], -1e18), float(ts[rr]))
            ut = Counter()
            for h in H[u]:
                v = vid[h["rows"]]
                for x, y in zip(v[:-1], v[1:]):
                    if x == vid[last]:
                        ut[y] += 1
            hh = hour[last]
            tod = Counter(vid[rr] for rr in hrows if min(abs(hour[rr] - hh), 24 - abs(hour[rr] - hh)) <= 2)
            d = np.hypot((P_lon - lon[last]) * np.cos(np.radians(lat[last])), P_lat - lat[last]) * 111.32
            dist = dict(zip(P_ids.tolist(), (-d).tolist()))
            ua = Counter(act[hrows].tolist()); tot_a = sum(ua.values()) + 1e-9
            apd = dict(zip(P_ids.tolist(), (np.log(np.array([ua.get(x, 0) + 0.5 for x in P_act]) / tot_a)
                                            - np.log1p(d)).tolist()))
            sigs = {"user frequency": dict(freq), "user recency": rec, "user transition": dict(ut),
                    "user time-of-day": dict(tod), "distance": dist, "activity prior x distance": apd,
                    "city popularity": dict(pop), "city transition": dict(trans.get(vid[last], {}))}
            for name, sc in sigs.items():
                if tgt in sc:
                    g, ties, _ = hit_probs(sc, tgt)
                else:   # target scored 0 -> tied with every unscored POI (and below positives)
                    vals = np.fromiter(sc.values(), float) if sc else np.zeros(0)
                    g = int((vals > 0).sum()); ties = len(P_ids) - len(sc) + int((vals == 0).sum())
                acc[name].append(score_metrics(g, max(ties, 1)))
            cover["target in user history"].append(tgt in freq)
            cover["target = a place of the current trip"].append(tgt in set(vid[rows_[:-1]].tolist()))
            cover["target within 1 km of current place"].append(-dist.get(tgt, -1e9) <= 1.0)
            cover["target among 100 nearest POIs"].append(int((d < -dist.get(tgt, -1e9)).sum()) < 100)
        r = {"city": pretty(c), "n": len(Q)}
        for name in acc:
            m = pd.DataFrame(acc[name]).mean()
            r[f"{name}|acc@1"] = m["acc@1"]; r[f"{name}|acc@5"] = m["acc@5"]; r[f"{name}|ndcg@5"] = m["ndcg@5"]
        for name, v in cover.items():
            r[f"cover|{name}"] = float(np.mean(v))
        per_city.append(r)
        print(f"[{c}] n={len(Q)} " + " ".join(f"{k.split('|')[0]}={v:.3f}" for k, v in r.items() if k.endswith("acc@1")))
    if not per_city:
        return
    P = pd.DataFrame(per_city)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    P.to_csv(os.path.join(a.out, "tables", "A4_information_value_per_city.csv"), index=False)
    sig_names = [k.split("|")[0] for k in P.columns if k.endswith("|acc@1")]
    rows = [{"signal": s, "knowledge": "user history" if s.startswith("user") else
             ("target-city log" if s.startswith("city") else "geometry / activity"),
             "used by GEMMA zero-shot": USED.get(s, ""), "macro Acc@1": P[f"{s}|acc@1"].mean(),
             "macro Acc@5": P[f"{s}|acc@5"].mean(), "macro NDCG@5": P[f"{s}|ndcg@5"].mean()} for s in sig_names]
    T = pd.DataFrame(rows).sort_values("macro Acc@1", ascending=False)
    save_table(T, a.out, "A4_information_value",
               "One-signal rankers on the STHGCN test protocol (macro over held-out cities). The user's own "
               "history carries most of the predictive information; GEMMA uses every signal that does not "
               "require target-city training data.")
    C2 = pd.DataFrame([{"coverage": k.split("|", 1)[1], "macro share": P[k].mean()} for k in P.columns if k.startswith("cover|")])
    save_table(C2, a.out, "A4_coverage", "Where the next place comes from (macro share of test targets).")
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    col = ["#0072B2" if "yes" in u else "#D55E00" for u in T["used by GEMMA zero-shot"]]
    ax.barh(T["signal"][::-1], T["macro Acc@1"][::-1], color=col[::-1])
    ax.set_xlabel("macro Acc@1 of a one-signal ranker")
    ax.set_title("blue: used by GEMMA zero-shot   orange: needs target-city data", fontsize=7)
    save_fig(fig, a.out, "A4a_information_value"); plt.close(fig)


if __name__ == "__main__":
    main()
