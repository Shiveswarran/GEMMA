"""
diagnose_caps.py -- model-free diagnostic of what the Massive-STEPS next-POI
task rewards, run on YOUR csv files (CPU only, a few minutes per city).

For every test trajectory (target = last check-in, context = the rest) it reports

  coverage (upper bounds on what a signal can ever retrieve)
    in_prefix      target already visited earlier in the SAME trajectory
    in_hist15      target in the user's 15 most-recent TRAIN trajectories
                   (exactly the history the LLM zero-shot baselines receive)
    in_hist_all    target anywhere in the user's train+validation history
    in_near100     target among the 100 POIs nearest to the last check-in
                   (= LLM-Move's candidate set)
    in_markov10    target among the top-10 train successors of the last POI
    in_union       any of the above

  model-free rankers over the FULL city vocabulary (all splits' POIs)
    user_freq      recency-weighted count in user history+prefix, geo tie-break
    markov         global POI->POI transition counts (train), popularity tie-break
    nearest        distance to last check-in
    combo          fixed-weight sum of the three + log popularity

  and the same rankers under the LLM zero-shot protocol (200 users x 1 random
  test trajectory, Acc@1/Acc@5/NDCG@5, several seeds) with three candidate sets:
    near100     100 POIs nearest the last check-in (truth may be absent)
    gt+near100  same, truth injected (replaces the farthest)
    gt+rand100  truth + 100 random POIs (the original LLM-Move paper protocol)

If `combo` already reaches the numbers of the SOTA tables, the task is mostly
history + transition + geography, and GEMMA must be GIVEN those inputs.

usage:
  python scripts/diagnose_caps.py --data_root data --cities tokyo istanbul ... \
      --out diag_caps.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SPLITS = ["train", "validation", "test"]


def load(root, city):
    fr = []
    for sp in SPLITS:
        p = os.path.join(root, city, f"{city}_checkins_{sp}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, usecols=lambda c: c in {
                "trail_id", "user_id", "venue_id", "latitude", "longitude",
                "venue_city_latitude", "venue_city_longitude", "timestamp",
                "venue_category_id"})
            d["_split"] = sp
            fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "venue_id"])
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    if "venue_city_latitude" in df:
        lat = lat.fillna(pd.to_numeric(df["venue_city_latitude"], errors="coerce"))
        lon = lon.fillna(pd.to_numeric(df["venue_city_longitude"], errors="coerce"))
    df["lat"] = lat.fillna(lat.mean()); df["lon"] = lon.fillna(lon.mean())
    df["venue_id"] = df["venue_id"].astype(np.int64)
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def dist_km(lat0, lon0, lat, lon):
    k = 111.32
    dx = (lon - lon0) * k * np.cos(np.radians(lat0))
    dy = (lat - lat0) * k
    return np.sqrt(dx * dx + dy * dy)


def rank_of(score, t):
    return int((score > score[t]).sum())


def metrics(ranks):
    r = np.asarray(ranks)
    return {"acc@1": float((r < 1).mean()), "acc@5": float((r < 5).mean()),
            "acc@10": float((r < 10).mean()),
            "ndcg@5": float(np.where(r < 5, 1 / np.log2(r + 2), 0).mean()),
            "mrr": float((1 / (r + 1)).mean())}


def activity_baselines(a, city, fsq):
    """Next-ACTIVITY baselines (no published numbers exist for this task):
    majority class, user's most frequent activity in history, global activity
    Markov chain (last activity -> next, from train), and user Markov."""
    df = load(a.data_root, city)
    df["act"] = [fsq.activity_id(c) for c in df["venue_category_id"].values]
    trails = {}; tpois = {}
    for (u, tid), g in df.groupby(["user_id", "trail_id"], sort=False):
        trails[(u, tid)] = (g["_split"].iloc[0], g["act"].values)
        tpois[(u, tid)] = g["venue_id"].values
    upois = defaultdict(set)
    for k, (sp, _) in trails.items():
        if sp in ("train", "validation"):
            upois[k[0]].update(tpois[k].tolist())
    A = int(df["act"].max()) + 1
    glob = np.zeros(A); M = np.zeros((A, A)); uh = defaultdict(lambda: np.zeros(A))
    uM = defaultdict(lambda: np.zeros((A, A)))
    for (u, tid), (sp, acts) in trails.items():
        if sp in ("train", "validation"):
            np.add.at(uh[u], acts, 1)
            for x, y in zip(acts[:-1], acts[1:]):
                uM[u][x, y] += 1
        if sp == "train":
            np.add.at(glob, acts, 1)
            for x, y in zip(acts[:-1], acts[1:]):
                M[x, y] += 1
    maj = int(glob.argmax())
    yt, preds = [], defaultdict(list)
    rev_by_act = {}
    for (u, tid), (sp, acts) in trails.items():
        if sp != "test" or len(acts) < 2:
            continue
        t, last = int(acts[-1]), int(acts[-2]); yt.append(t)
        pv = tpois[(u, tid)]
        ret = (pv[-1] in upois[u]) or (pv[-1] in set(pv[:-1].tolist()))
        rev_by_act.setdefault(t, []).append(int(ret))
        h = uh[u] + 0; h_pref = np.bincount(acts[:-1], minlength=A)
        preds["majority"].append(maj)
        preds["user_freq"].append(int((h + h_pref + 1e-3 * glob).argmax()))
        preds["markov"].append(int((M[last] + 1e-3 * glob).argmax()))
        um = uM[u][last]
        preds["user_markov"].append(int((um + 0.5 * M[last] / max(1, M[last].sum())
                                         + 1e-3 * glob).argmax()))
        sc = (np.log1p(h + h_pref) + np.log1p(M[last] / max(1, M[last].sum()) * 10)
              + np.log1p(um))
        preds["combo"].append(int(sc.argmax()))
    yt = np.asarray(yt); out = {}
    for k, p in preds.items():
        p = np.asarray(p); f1 = []
        for c in np.unique(np.r_[yt, p]):
            tp = ((p == c) & (yt == c)).sum(); fp = ((p == c) & (yt != c)).sum()
            fn = ((p != c) & (yt == c)).sum()
            pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
            f1.append(2 * pr * rc / (pr + rc + 1e-9))
        out[k] = {"acc": round(float((p == yt).mean()), 4), "macro_f1": round(float(np.mean(f1)), 4)}
    names = {}
    try:
        names = {v: k for k, v in fsq.act2id.items()}
    except Exception:
        pass
    out["revisit_rate_by_target_activity"] = {
        str(names.get(a, a)): {"n": len(v), "revisit": round(float(np.mean(v)), 3)}
        for a, v in sorted(rev_by_act.items(), key=lambda x: -len(x[1]))}
    print(f"[{city}] revisit rate by target activity: " + "  ".join(
        f"{k[:18]}={v['revisit']:.2f}(n={v['n']})" for k, v in out["revisit_rate_by_target_activity"].items()))
    print(f"\n[{city}] next-ACTIVITY baselines (n={len(yt)}): " +
          "  ".join(f"{k} acc={v['acc']:.3f}/f1={v['macro_f1']:.3f}" for k, v in out.items()
                    if isinstance(v, dict) and "acc" in v))
    sys.stdout.flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--max_queries", type=int, default=5000,
                    help="subsample test trajectories for the full-vocab rankers")
    ap.add_argument("--hist_k", type=int, default=15)
    ap.add_argument("--llm_users", type=int, default=200)
    ap.add_argument("--llm_seeds", type=int, default=5)
    ap.add_argument("--out", default="diag_caps.json")
    ap.add_argument("--fsq_csv", default=None,
                    help="taxonomy CSV -> also report next-ACTIVITY baselines")
    ap.add_argument("--skip_poi", action="store_true", help="only next-activity baselines")
    a = ap.parse_args()

    report = {}
    fsq = None
    if a.fsq_csv:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gemma.fsq import FSQTaxonomy
        fsq = FSQTaxonomy(a.fsq_csv)
    for city in a.cities:
        if fsq is not None:
            report.setdefault(city, {})["next_activity"] = activity_baselines(a, city, fsq)
            if a.skip_poi:
                continue
        df = load(a.data_root, city)
        # ---- vocabulary = all POIs of the city (all splits), as in the benchmark
        pois = df.groupby("venue_id").agg(lat=("lat", "mean"), lon=("lon", "mean"))
        vids = pois.index.values; P = len(vids)
        v2i = {int(v): i for i, v in enumerate(vids)}
        plat = pois["lat"].values; plon = pois["lon"].values
        df["pi"] = df["venue_id"].map(v2i).astype(np.int64)

        # ---- trajectories
        trails = {}
        for (u, tid), g in df.groupby(["user_id", "trail_id"], sort=False):
            trails[(u, tid)] = (g["_split"].iloc[0], g["pi"].values, g["timestamp"].iloc[0])
        hist_trails = defaultdict(list)           # user -> [(start, pis)] train+val
        for (u, tid), (sp, pis, t0) in trails.items():
            if sp in ("train", "validation"):
                hist_trails[u].append((t0, sp, pis))
        for u in hist_trails:
            hist_trails[u].sort(key=lambda x: x[0])
        # train statistics
        pop = np.zeros(P); trans = defaultdict(Counter)
        for (u, tid), (sp, pis, t0) in trails.items():
            if sp != "train":
                continue
            np.add.at(pop, pis, 1)
            for x, y in zip(pis[:-1], pis[1:]):
                trans[int(x)][int(y)] += 1
        logpop = np.log1p(pop)

        test = [(u, pis) for (u, tid), (sp, pis, t0) in trails.items()
                if sp == "test" and len(pis) >= 2]
        test_vocab = len(set(df.loc[df._split == "test", "pi"]))
        rng = np.random.RandomState(0)

        def features(u, pis):
            prefix, tgt, last = pis[:-1], int(pis[-1]), int(pis[-2])
            ht = [x for x in hist_trails.get(u, []) if x[1] == "train"][-a.hist_k:]
            h15 = set(np.concatenate([x[2] for x in ht])) if ht else set()
            hall = set(np.concatenate([x[2] for x in hist_trails[u]])) if hist_trails.get(u) else set()
            d = dist_km(plat[last], plon[last], plat, plon)
            freq = np.zeros(P)
            seq = [x[2] for x in hist_trails.get(u, [])] + [prefix]
            n = len(seq)
            for j, s in enumerate(seq):
                np.add.at(freq, s, 0.9 ** (n - 1 - j))
            mk = np.zeros(P)
            for y, c in trans.get(last, {}).items():
                mk[y] = c
            return prefix, tgt, last, h15, hall, d, freq, mk

        cov = Counter(); nq = 0
        ranks = defaultdict(list)
        q_idx = rng.permutation(len(test))[:a.max_queries]
        n_pref = []; n_hist = []
        for qi in q_idx:
            u, pis = test[qi]
            prefix, tgt, last, h15, hall, d, freq, mk = features(u, pis)
            near = set(np.argpartition(d, min(100, P - 1))[:100])
            top_mk = set(np.argsort(-mk)[:10]) if mk.max() > 0 else set()
            c = {"in_prefix": tgt in set(prefix), "in_hist15": tgt in h15,
                 "in_hist_all": tgt in hall, "in_near100": tgt in near,
                 "in_markov10": tgt in top_mk}
            c["in_union"] = any(c.values())
            for k, v in c.items():
                cov[k] += int(v)
            nq += 1
            n_pref.append(len(prefix)); n_hist.append(len(hist_trails.get(u, [])))
            geo = -np.log1p(d)
            s_user = np.log1p(freq) * 10 + geo * 1e-3
            s_mk = np.log1p(mk) * 10 + logpop * 1e-3
            s_combo = 3 * np.log1p(freq) + 2 * np.log1p(mk) + 1.0 * geo + 0.5 * logpop
            ranks["user_freq"].append(rank_of(s_user, tgt))
            ranks["markov"].append(rank_of(s_mk, tgt))
            ranks["nearest"].append(rank_of(geo, tgt))
            ranks["combo"].append(rank_of(s_combo, tgt))

        # ---- LLM zero-shot protocol (200 users x 1 traj, 100 nearest candidates)
        by_user = defaultdict(list)
        for i, (u, pis) in enumerate(test):
            by_user[u].append(i)
        users = list(by_user)
        llm = {v: defaultdict(list) for v in ("near100", "gt+near100", "gt+rand100")}
        for seed in range(a.llm_seeds):
            r2 = np.random.RandomState(seed)
            sel = r2.choice(len(users), min(a.llm_users, len(users)), replace=False)
            rr = {v: defaultdict(list) for v in llm}
            for ui in sel:
                u = users[ui]; i = by_user[u][r2.randint(len(by_user[u]))]
                _, pis = test[i]
                prefix, tgt, last, h15, hall, d, freq, mk = features(u, pis)
                near = np.argsort(d)[:100]
                variants = {"near100": near}
                g = near.copy()
                if tgt not in set(g):
                    g[-1] = tgt                          # inject truth (replace farthest)
                variants["gt+near100"] = g
                neg = r2.choice(np.setdiff1d(np.arange(P), [tgt]), min(100, P - 1), replace=False)
                variants["gt+rand100"] = np.r_[neg, tgt]  # LLM-Move paper: truth + 100 random
                for vname, cand in variants.items():
                    if tgt not in set(cand):
                        for k in ("user_freq", "markov", "nearest", "combo"):
                            rr[vname][k].append(10**6)
                        continue
                    geo = -np.log1p(d[cand])
                    sc = {"user_freq": np.log1p(freq[cand]) * 10 + geo * 1e-3,
                          "markov": np.log1p(mk[cand]) * 10 + logpop[cand] * 1e-3,
                          "nearest": geo,
                          "combo": 3 * np.log1p(freq[cand]) + 2 * np.log1p(mk[cand]) + geo + 0.5 * logpop[cand]}
                    t = int(np.where(cand == tgt)[0][0])
                    for k, s_ in sc.items():
                        rr[vname][k].append(rank_of(s_, t))
            for vname in rr:
                for k, v in rr[vname].items():
                    m = metrics(v); llm[vname][k].append([m["acc@1"], m["acc@5"], m["ndcg@5"]])
        llm_sum = {vname: {k: {"acc@1": f"{np.mean(v, 0)[0]:.3f}+-{np.std(v, 0)[0]:.3f}",
                               "acc@5": f"{np.mean(v, 0)[1]:.3f}+-{np.std(v, 0)[1]:.3f}",
                               "ndcg@5": f"{np.mean(v, 0)[2]:.3f}+-{np.std(v, 0)[2]:.3f}"}
                           for k, v in d_.items()} for vname, d_ in llm.items()}

        rep = {"n_pois_all_splits": P, "n_pois_test_split": test_vocab,
               "n_test_traj": len(test), "n_queries_used": nq,
               "mean_prefix_len": float(np.mean(n_pref)),
               "mean_hist_trajectories": float(np.mean(n_hist)),
               "coverage": {k: round(v / max(1, nq), 4) for k, v in cov.items()},
               "full_vocab_rankers": {k: {kk: round(vv, 4) for kk, vv in metrics(v).items()}
                                      for k, v in ranks.items()},
               "llm_protocol_rankers": llm_sum}
        report[city] = {**report.get(city, {}), **rep}
        print(f"\n===== {city} =====  POIs(all)={P}  POIs(test-only)={test_vocab}  "
              f"test traj={len(test)}  prefix_len={rep['mean_prefix_len']:.2f}  "
              f"hist_traj={rep['mean_hist_trajectories']:.1f}")
        print("coverage:", rep["coverage"])
        for k, m in rep["full_vocab_rankers"].items():
            print(f"  full-vocab {k:<10} acc@1={m['acc@1']:.4f} acc@5={m['acc@5']:.4f} "
                  f"acc@10={m['acc@10']:.4f} ndcg@5={m['ndcg@5']:.4f} mrr={m['mrr']:.4f}")
        for vname, d_ in llm_sum.items():
            for k, m in d_.items():
                print(f"  LLM[{vname:<10}] {k:<10} acc@1={m['acc@1']} acc@5={m['acc@5']} ndcg@5={m['ndcg@5']}")
        sys.stdout.flush()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
