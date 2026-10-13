"""A3 — Why activity-based pretraining is strong: daily activity sequences are LOW-dimensional,
CONSISTENT within users and SHARED across cities; POI-level days are none of these.

Unit of analysis: a user-day (all check-ins of one user on one calendar day). Each user-day is
encoded at three levels of abstraction (rows L2-normalised):
  POI         bag of venue ids                           (city-specific vocabulary)
  category    bag of leaf Foursquare categories          (shared vocabulary, fine)
  activity    activity x hour-of-day occupancy (A x 24)  (shared, coarse)  <- GEMMA's unit
Analyses
  (1) PCA / SVD energy: how many dimensions explain 50 / 80 % of the variance of the
      user-day matrix, per level and city (low-rank = learnable, compressible structure)
  (2) cosine similarity between user-days: same user, different users same city,
      different cities  (POI-level cross-city similarity is exactly 0)
  (3) subspace transfer: share of a TARGET city's activity-day variance explained by the
      top-k principal components fitted on the SOURCE cities only
  (4) "eigen-days": the leading source principal components drawn as activity x hour maps
Outputs: tables A3_pca_rank, A3_cosine, A3_subspace_transfer; figures A3a_scree, A3b_cosine,
         A3c_subspace_transfer, A3d_eigendays
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from common import (SRC, TGT, base_args, load_cfg, taxonomy, load_city, save_table, setup_mpl, save_fig,
                    pretty, city_style, short_act)


def user_days(C, A, cat_index, max_days, rng):
    df = C["df"]
    keys = df.groupby(["user_id", "date"]).indices
    items = list(keys.items())
    if len(items) > max_days:
        items = [items[i] for i in rng.choice(len(items), max_days, replace=False)]
    vids = df["venue_id"].values; acts = df["act"].values; hours = df["hour"].values
    cats = df["venue_category_id"].astype(str).values
    poi_vocab = {}
    poi_rows, poi_cols, cat_rows, cat_cols = [], [], [], []
    X_act = np.zeros((len(items), A * 24), np.float32)
    users = []
    for i, ((u, _), idx) in enumerate(items):
        users.append(u)
        for r in idx:
            poi_rows.append(i); poi_cols.append(poi_vocab.setdefault(vids[r], len(poi_vocab)))
            if cats[r] in cat_index:
                cat_rows.append(i); cat_cols.append(cat_index[cats[r]])
            X_act[i, acts[r] * 24 + hours[r]] += 1
    def sparse(rows, cols, ncol):
        M = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.ones(len(rows)), (len(items), ncol)).coalesce()
        return M
    P = sparse(poi_rows, poi_cols, max(1, len(poi_vocab)))
    Cm = sparse(cat_rows, cat_cols, max(1, len(cat_index)))
    return {"users": np.array(users), "poi": P, "cat": Cm, "act": torch.tensor(X_act), "poi_vocab": poi_vocab}


def row_normalise(M):
    if M.is_sparse:
        M = M.coalesce(); i = M.indices(); v = M.values()
        n = torch.zeros(M.size(0)).index_add_(0, i[0], v * v).sqrt().clamp_min(1e-9)
        return torch.sparse_coo_tensor(i, v / n[i[0]], M.shape).coalesce()
    return M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)


def energy_curve(M, q=50):
    """Fraction of total (uncentred) energy captured by the top-k singular directions."""
    M = row_normalise(M)
    total = float((M.coalesce().values() ** 2).sum()) if M.is_sparse else float((M ** 2).sum())
    q = int(min(q, M.shape[0] - 1, M.shape[1] - 1))
    if q < 1:
        return np.array([1.0])
    _, S, _ = torch.svd_lowrank(M, q=q, niter=4)
    return np.cumsum((S.numpy() ** 2)) / max(total, 1e-9)


def k_for(curve, frac):
    hit = np.where(curve >= frac)[0]
    return int(hit[0] + 1) if len(hit) else f">{len(curve)}"


def mean_pair_cos(Xa, Xb, pairs):
    if len(pairs) == 0:
        return float("nan")
    i, j = np.array(pairs).T
    if Xa.is_sparse:
        Xa = Xa.to_dense() if Xa.shape[1] < 20000 else None
    if Xa is None:
        return float("nan")
    a = Xa[torch.tensor(i)]; b = Xb[torch.tensor(j)]
    return float((a * b).sum(1).mean())


def sparse_rows_cos(M, pairs):
    """cosine for sparse rows without densifying the whole matrix."""
    M = row_normalise(M).coalesce()
    idx = M.indices().numpy(); val = M.values().numpy()
    rows = {}
    for r, c, v in zip(idx[0], idx[1], val):
        rows.setdefault(int(r), {})[int(c)] = float(v)
    out = []
    for i, j in pairs:
        a, b = rows.get(i, {}), rows.get(j, {})
        if len(a) > len(b): a, b = b, a
        out.append(sum(v * b.get(k, 0.0) for k, v in a.items()))
    return float(np.mean(out)) if out else float("nan")


def main():
    ap = base_args(__doc__)
    ap.add_argument("--max_days", type=int, default=40000)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--k_transfer", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    cfg = load_cfg(a); fsq, names = taxonomy(cfg)
    A = max(fsq.act2id.values()) + 1
    cat_index = {c: i for i, c in enumerate(sorted(fsq.id2levels.keys()))}
    plt = setup_mpl()
    U = {}
    for c in a.cities:
        try:
            C = load_city(cfg, fsq, c)
        except FileNotFoundError:
            print(f"skip {c}"); continue
        U[c] = user_days(C, A, cat_index, a.max_days, rng)
        print(f"[{c}] user-days={len(U[c]['users'])} POIs={len(U[c]['poi_vocab'])}")
    cities = list(U)
    # (1) PCA rank
    rows, curves = [], {}
    for c in cities:
        r = {"city": pretty(c), "user-days": len(U[c]["users"])}
        for lev in ("poi", "cat", "act"):
            cv = energy_curve(U[c][lev]); curves[(c, lev)] = cv
            r[f"{lev}: dims"] = U[c][lev].shape[1]
            r[f"{lev}: k@50%"] = k_for(cv, 0.5); r[f"{lev}: k@80%"] = k_for(cv, 0.8)
        rows.append(r)
    save_table(pd.DataFrame(rows), a.out, "A3_pca_rank",
               "Number of principal directions needed to explain 50\\%/80\\% of the user-day variance. "
               "Activity-level days are low-rank; POI-level days are not.")
    # (2) cosine similarity
    rows = []
    for c in cities:
        users = U[c]["users"]; n = len(users)
        by_u = pd.Series(np.arange(n)).groupby(users).apply(list)
        same = []
        for lst in by_u:
            if len(lst) >= 2:
                k = rng.choice(len(lst), 2, replace=False); same.append((lst[k[0]], lst[k[1]]))
        same = same[:a.pairs]
        diff = [(int(i), int(j)) for i, j in rng.randint(0, n, size=(a.pairs, 2)) if users[i] != users[j]]
        r = {"city": pretty(c)}
        for lev in ("poi", "cat", "act"):
            M = U[c][lev]
            f = (lambda pr: sparse_rows_cos(M, pr)) if M.is_sparse else (lambda pr: mean_pair_cos(row_normalise(M), row_normalise(M), pr))
            r[f"{lev}: same user"] = f(same); r[f"{lev}: other users"] = f(diff)
        rows.append(r)
    cross = {"poi": [], "cat": [], "act": []}
    for i, ci in enumerate(cities):
        for cj in cities[i + 1:]:
            ni, nj = len(U[ci]["users"]), len(U[cj]["users"])
            pr = list(zip(rng.randint(0, ni, a.pairs // 4), rng.randint(0, nj, a.pairs // 4)))
            cross["poi"].append(0.0)                      # disjoint POI vocabularies
            Xi = row_normalise(U[ci]["act"]); Xj = row_normalise(U[cj]["act"])
            cross["act"].append(mean_pair_cos(Xi, Xj, pr))
            Ci = row_normalise(U[ci]["cat"]).to_dense(); Cj = row_normalise(U[cj]["cat"]).to_dense()
            cross["cat"].append(mean_pair_cos(Ci, Cj, pr))
    T = pd.DataFrame(rows)
    mean_row = {"city": "mean"}
    for col in T.columns[1:]:
        mean_row[col] = float(np.nanmean(T[col].astype(float)))
    T = pd.concat([T, pd.DataFrame([mean_row])], ignore_index=True)
    for lev in cross:
        T[f"{lev}: other city"] = [float("nan")] * (len(T) - 1) + [float(np.mean(cross[lev])) if cross[lev] else float("nan")]
    save_table(T, a.out, "A3_cosine",
               "Mean cosine similarity between user-days. POI-level days of different users (and of "
               "different cities, by construction) are orthogonal; activity-level days are similar "
               "within users, across users and across cities.")
    # (3) subspace transfer
    src = [c for c in cities if c in SRC] or cities[: max(1, len(cities) // 2)]
    tgt = [c for c in cities if c not in src]
    rows = []
    if src and tgt:
        Xs = torch.cat([row_normalise(U[c]["act"]) for c in src]); mu = Xs.mean(0, keepdim=True)
        _, _, V = torch.linalg.svd(Xs - mu, full_matrices=False)
        k = a.k_transfer; Vs = V[:k].T
        for c in tgt:
            Xt = row_normalise(U[c]["act"]); Xt = Xt - Xt.mean(0, keepdim=True)
            tot = float((Xt ** 2).sum())
            own = torch.linalg.svd(Xt, full_matrices=False)[1][:k]
            own_var = float((own ** 2).sum()) / tot
            src_var = float(((Xt @ Vs) ** 2).sum()) / tot
            rows.append({"target city": pretty(c), f"own top-{k} PCs": own_var,
                         f"source top-{k} PCs": src_var, "transfer ratio": src_var / max(own_var, 1e-9)})
        save_table(pd.DataFrame(rows), a.out, "A3_subspace_transfer",
                   f"Share of a held-out city's activity-day variance explained by its own top-{k} principal "
                   f"components vs. by the top-{k} components fitted on the source cities only.")
    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    cmap = plt.get_cmap("tab20")
    for ax, lev, title in zip(axes, ("poi", "cat", "act"), ("POI (venue id)", "category", "activity × hour")):
        for i, c in enumerate(cities):
            cv = curves[(c, lev)]
            ax.plot(np.arange(1, len(cv) + 1), cv, color=cmap(i % 20), label=pretty(c), **city_style(c))
        ax.set_title(title); ax.set_xlabel("principal directions $k$"); ax.axhline(0.8, color="k", lw=0.5, ls=":")
    axes[0].set_ylabel("variance explained"); axes[-1].legend(fontsize=5, frameon=False, ncol=2)
    save_fig(fig, a.out, "A3a_scree"); plt.close(fig)
    m = T.iloc[-1]
    groups = ["same user", "other users", "other city"]
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    w = 0.26
    for li, (lev, lab, col) in enumerate((("poi", "POI", "#999999"), ("cat", "category", "#56B4E9"), ("act", "activity×hour", "#0072B2"))):
        vals = [m.get(f"{lev}: {g}", float("nan")) for g in groups]
        ax.bar(np.arange(3) + (li - 1) * w, vals, w, label=lab, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels(groups); ax.set_ylabel("mean cosine of user-days")
    ax.legend(frameon=False)
    save_fig(fig, a.out, "A3b_cosine"); plt.close(fig)
    if rows:
        R = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(3.6, 2.2))
        x = np.arange(len(R))
        ax.bar(x - 0.2, R.iloc[:, 1], 0.4, label="own PCs", color="#999999")
        ax.bar(x + 0.2, R.iloc[:, 2], 0.4, label="source-city PCs", color="#0072B2")
        ax.set_xticks(x); ax.set_xticklabels(R["target city"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("variance explained"); ax.legend(frameon=False)
        save_fig(fig, a.out, "A3c_subspace_transfer"); plt.close(fig)
        acts = [k for k in range(1, A) if float(Xs[:, k * 24:(k + 1) * 24].sum()) > 0]
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.22 * len(acts) + 0.9), sharey=True)
        for p, ax in enumerate(axes):
            W = V[p].reshape(A, 24).numpy()[acts]; lim = np.abs(W).max()
            im = ax.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_title(f"eigen-day {p + 1}"); ax.set_xlabel("hour"); ax.set_xticks([0, 6, 12, 18, 23])
        axes[0].set_yticks(range(len(acts))); axes[0].set_yticklabels([short_act(names.get(k, str(k))) for k in acts], fontsize=6)
        save_fig(fig, a.out, "A3d_eigendays"); plt.close(fig)


if __name__ == "__main__":
    main()
