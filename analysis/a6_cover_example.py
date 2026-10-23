"""A6 — Real examples for the cover figure (Fig. 1) of the paper.

(a) "How GEMMA predicts in a city it has never been trained on": one real test query from a
    held-out city that GEMMA ranks correctly at top-1.
      left   user's history places (relative km, jittered; size = visits, colour = activity),
             the current trip (arrows) and the true next place (star)
      middle the user's history as a day x hour raster coloured by activity (routines)
      right  GEMMA's top-5 next places with probabilities, split into return / explore,
             and the evidence behind them (visits, days since last visit, visits at this hour)
(b) "Activity is the transferable unit": two real user-days from two different cities whose
    activity-by-hour patterns are almost identical while they share no place at all.

Privacy: user ids are never shown, coordinates are shifted to the current place and jittered
(+-200 m by default), no POI names unless --show_names.  Check your data licence before
publishing any real trajectory.

  python analysis/a6_cover_example.py --run_dir runs/gemma_v4 --city tokyo --pair_cities istanbul tokyo
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from common import (SRC, TGT, base_args, load_cfg, taxonomy, load_city, setup_mpl, save_fig, pretty,
                    ACT_COLORS, save_table, short_act)


def haversine_km(lat, lon, i, j):
    p1, p2 = np.radians(lat[i]), np.radians(lat[j])
    dl = np.radians(lon[j] - lon[i])
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(min(1.0, h))))


def pick_query(bb, head, ci, qs, cfg, rng, want="revisit", min_hist=8, bs=64):
    from gemma.realize import _forward, _amp
    order = rng.permutation(len(qs))
    best = (10 ** 9, None, None)          # fallback: best-ranked query meeting the history criteria
    for s in range(0, len(order), bs):
        sub = [qs[i] for i in order[s:s + bs]]
        with torch.no_grad():
            o = _forward(bb, head, ci, sub, False, _amp(cfg, ci.device), 0)
        lg = o["logits"].float(); tgt = o["m"]["tgt_pi"]; ar = torch.arange(len(sub), device=tgt.device)
        rank = (lg > lg.gather(1, tgt[:, None])).sum(1)
        new = ~o["visited"][ar, tgt]
        for b in range(len(sub)):
            ti, j = sub[b]
            hs = ci.history(ti)
            n_hist = sum(len(ci.trails[h]["rows"]) for h in hs)
            acts = {int(ci.act_row[r]) for h in hs for r in ci.trails[h]["rows"]}
            ok_kind = bool(new[b]) if want == "explore" else (not bool(new[b]))
            if ok_kind and n_hist >= min_hist and len(acts) >= 3 and j >= 2:
                sel = {k: (v[b] if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == len(sub) else v)
                       for k, v in o.items() if k in ("logits", "act_logits", "gate", "visited")}
                if rank[b] == 0:
                    return sub[b], sel
                if int(rank[b]) < best[0]:
                    best = (int(rank[b]), sub[b], sel)
    if best[1] is not None:
        print(f"no top-1 hit met the criteria; using the best-ranked query (rank {best[0] + 1})")
    return best[1], best[2]


def cover_query(a, cfg, fsq, names, plt, rng):
    from gemma.poi_encoder import POIEncoder
    from gemma.utils import check_device, get_logger
    from gemma.realize import sthgcn_filter
    from a5_learned_behaviour import load_model
    logger = get_logger("gemma.a6"); cfg.allow_cpu = a.allow_cpu
    device = check_device(cfg, logger)
    enc = POIEncoder(cfg, logger); cfg.poi_emb_dim = enc.dim
    bb, head, ci, ids = load_model(cfg, a.run_dir, a.city, device, fsq, enc, logger)
    qs = sthgcn_filter(ci, ci.queries(ids["test"], False), ids["train"])
    q, o = pick_query(bb, head, ci, qs, cfg, rng, a.want)
    if q is None:
        print("no suitable query found"); return
    ti, j = q
    tr = ci.trails[ti]; rows = tr["rows"]; prefix = rows[:j]; tgt_row = rows[j]
    lat = ci.lat.cpu().numpy(); lon = ci.lon.cpu().numpy()
    pi_of = lambda r: int(ci.row_pi[r])
    cur = pi_of(prefix[-1]); tgt = pi_of(tgt_row)
    lat0, lon0 = lat[cur], lon[cur]
    jit = {}
    def xy(p):
        if p not in jit:
            jit[p] = rng.uniform(-a.jitter_km, a.jitter_km, 2)
        return ((lon[p] - lon0) * 111.32 * np.cos(np.radians(lat0)) + jit[p][0], (lat[p] - lat0) * 111.32 + jit[p][1])
    hs = ci.history(ti)
    hrows = [r for h in hs for r in ci.trails[h]["rows"]]
    visits = pd.Series([pi_of(r) for r in hrows]).value_counts()
    poi_act = {pi_of(r): int(ci.act_row[r]) for r in list(hrows) + list(rows)}
    logits = o["logits"]; P = logits.exp().cpu().numpy()
    top = np.argsort(-P)[:5]
    visited = o["visited"].cpu().numpy()
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    for p, n in visits.items():
        x, y = xy(p)
        ax.scatter(x, y, s=10 + 18 * n, color=ACT_COLORS[poi_act.get(p, 0) % len(ACT_COLORS)], alpha=0.75,
                   edgecolor="white", lw=0.4, zorder=2)
    px = [xy(pi_of(r)) for r in prefix]
    for (x1, y1), (x2, y2) in zip(px[:-1], px[1:]):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.scatter(*px[-1], marker="o", s=40, facecolor="none", edgecolor="k", lw=1.2, zorder=3)
    tx, ty = xy(tgt)
    ax.scatter(tx, ty, marker="*", s=160, color="#D55E00", edgecolor="k", lw=0.5, zorder=4)
    for k, p in enumerate(top[:3]):
        x, y = xy(int(p)); ax.annotate(f"#{k + 1}", (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("km east of current place"); ax.set_ylabel("km north")
    ax.set_title(f"{pretty(a.city)} (never seen in training)", fontsize=8)
    ax2 = fig.add_subplot(gs[1])
    days = sorted({float(np.floor(ci.day_row[r])) for r in hrows})
    dmap = {d: i for i, d in enumerate(days)}
    for r in hrows:
        p = pi_of(r)
        ax2.scatter(int(ci.hour_row[r]), dmap[float(np.floor(ci.day_row[r]))], s=26 if p == tgt else 14,
                    color=ACT_COLORS[int(ci.act_row[r]) % len(ACT_COLORS)],
                    edgecolor="k" if p == tgt else "none", lw=0.8)
    ax2.set_xlim(-0.5, 23.5); ax2.set_xticks([0, 6, 12, 18, 23]); ax2.set_xlabel("hour of day")
    ax2.set_ylabel("history day"); ax2.invert_yaxis()
    ax2.set_title("history (outlined = true next place)", fontsize=8)
    used = sorted({int(ci.act_row[r]) for r in hrows})
    for k in used:
        ax2.scatter([], [], color=ACT_COLORS[k % len(ACT_COLORS)], s=12, label=short_act(names.get(k, str(k))))
    h2, l2 = ax2.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h2 += [Line2D([], [], marker="*", ls="", color="#D55E00", mec="k", mew=0.5, ms=8),
           Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=5)]
    l2 += ["true next place", "current place"]
    fig.legend(h2, l2, fontsize=6, frameon=False, loc="lower center", ncol=min(len(l2), 6),
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    ax3 = fig.add_subplot(gs[2])
    hh = int(ci.hour_row[prefix[-1]]); dnow = float(ci.day_row[prefix[-1]])
    labels, vals, cols, ev = [], [], [], []
    for k, p in enumerate(top):
        p = int(p)
        hv = [r for r in hrows if pi_of(r) == p]
        past = [dnow - float(ci.day_row[r]) for r in hv if ci.day_row[r] <= dnow]
        at_h = sum(1 for r in hv if min(abs(int(ci.hour_row[r]) - hh), 24 - abs(int(ci.hour_row[r]) - hh)) <= 2)
        nm = short_act(names.get(poi_act.get(p, int(ci.act[p])), "?"))
        if hv:
            when = f"last {min(past):.0f} d ago" if past else "seen in history"
            e = f"{nm}: {len(hv)}× visited, {when}" + (f", {at_h}× near {hh}:00" if at_h else "")
        else:
            e = f"{nm}: new place, {haversine_km(lat, lon, cur, p):.1f} km away"
        labels.append(f"#{k + 1}" + (" ★" if p == tgt else ""))
        vals.append(P[p]); cols.append("#0072B2" if visited[p] else "#E69F00"); ev.append(e)
    yy = np.arange(len(top))[::-1]
    ax3.barh(yy, vals, color=cols, height=0.38)
    for y, e in zip(yy, ev):
        ax3.text(0, y + 0.27, e, fontsize=5.2, color="#222222", va="bottom", ha="left")
    ax3.set_yticks(yy); ax3.set_yticklabels(labels, fontsize=7)
    for t, lab in zip(ax3.get_yticklabels(), labels):
        if "★" in lab:
            t.set_color("#D55E00"); t.set_fontweight("bold")
    ax3.set_xlim(0, max(vals) * 1.15); ax3.set_ylim(-0.5, len(top) - 0.1)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_xlabel("GEMMA probability")
    ax3.set_title("top-5: blue = return, orange = explore", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27)
    save_fig(fig, a.out, f"A6a_cover_{a.city}"); plt.close(fig)
    pa = o["act_logits"].exp().cpu().numpy(); topa = np.argsort(-pa)[:3]
    print("next-activity:", [(short_act(names.get(int(k), k)), round(float(pa[k]), 3)) for k in topa])


def transferable_pair(a, cfg, fsq, names, plt, rng):
    ca, cb = a.pair_cities
    vec = {}
    for c in (ca, cb):
        C = load_city(cfg, fsq, c); df = C["df"]
        A = max(fsq.act2id.values()) + 1
        items = [(k, idx) for k, idx in df.groupby(["user_id", "date"]).indices.items()
                 if len(idx) >= 3 and df["act"].values[idx].tolist().count(df["act"].values[idx][0]) < len(idx)]
        if len(items) > a.pair_pool:
            items = [items[i] for i in rng.choice(len(items), a.pair_pool, replace=False)]
        X = np.zeros((len(items), A * 24), np.float32)
        for i, (_, idx) in enumerate(items):
            for r in idx:
                X[i, df["act"].values[r] * 24 + df["hour"].values[r]] += 1
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        vec[c] = (X, items, df)
    Xa, Ia, Da = vec[ca]; Xb, Ib, Db = vec[cb]
    S = Xa @ Xb.T; i, j = np.unravel_index(np.argmax(S), S.shape)
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 1.9), sharex=True)
    for ax, (c, (_, idx), D) in zip(axes, ((ca, Ia[i], Da), (cb, Ib[j], Db))):
        for n, r in enumerate(idx):
            k = int(D["act"].values[r]); h = D["timestamp"].iloc[r]
            hx = h.hour + h.minute / 60
            ax.barh(0, 0.9, left=hx, color=ACT_COLORS[k % len(ACT_COLORS)])
            ax.annotate(short_act(names.get(k, str(k))), (hx, 0.48), fontsize=5.5, rotation=30)
        ax.set_yticks([]); ax.set_ylabel(pretty(c), rotation=0, ha="right", va="center", fontsize=8)
        ax.set_ylim(-0.6, 1.4)
    axes[-1].set_xlim(0, 24); axes[-1].set_xticks([0, 6, 12, 18, 24]); axes[-1].set_xlabel("hour of day")
    shared = len(set(Da["venue_id"].values[Ia[i][1]]) & set(Db["venue_id"].values[Ib[j][1]]))
    fig.suptitle(f"two real user-days: activity cosine {S[i, j]:.2f}, shared places = {shared}", fontsize=8)
    save_fig(fig, a.out, f"A6b_transferable_unit_{ca}_{cb}"); plt.close(fig)


def main():
    ap = base_args(__doc__)
    ap.add_argument("--city", default="tokyo")
    ap.add_argument("--want", choices=["revisit", "explore"], default="revisit")
    ap.add_argument("--pair_cities", nargs=2, default=["istanbul", "tokyo"])
    ap.add_argument("--pair_pool", type=int, default=5000)
    ap.add_argument("--jitter_km", type=float, default=0.2)
    ap.add_argument("--show_names", action="store_true")
    ap.add_argument("--no_model", action="store_true", help="only the transferable-unit figure")
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    cfg = load_cfg(a); fsq, names = taxonomy(cfg)
    plt = setup_mpl()
    if not a.no_model:
        cover_query(a, cfg, fsq, names, plt, rng)
    transferable_pair(a, cfg, fsq, names, plt, rng)


if __name__ == "__main__":
    main()
