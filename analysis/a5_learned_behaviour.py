"""A5 — What user behaviour did GEMMA learn, and what does the transformer encode?

(a) Learned behavioural "laws": the source-trained evidence weights of the return / explore
    channels, evaluated per test query in the held-out cities (mean, 10th-90th pct). In the
    log-linear return channel exp(r_v) = prod_k f_k^w_k, each weight is an exponent, e.g.
    (1+visits)^w_hist = preferential return, exp(w_rec * e^{-days/14}) = recency, etc.
(b) Return/explore gate calibration: predicted P(explore) for the target's activity vs the
    observed share of new places, per activity and as a reliability diagram; and P(explore)
    vs number of distinct places S (the EPR novelty law, learned without supervision on S).
(c) Representation probes on the frozen backbone latent z (no fine-tuning):
    linear next-activity and revisit/explore probes trained on SOURCE cities, tested on
    held-out cities, for the JEPA-pretrained backbone vs a backbone without JEPA pretraining.
    Plus PCA (variance explained) of z and a 2-D projection coloured by next activity.
Outputs: tables A5_learned_weights, A5_gate_calibration, A5_probes; figures A5a_gate_by_activity,
         A5b_reliability, A5c_epr_learned, A5d_latent_pca
Requires the trained checkpoints (GPU recommended):
  python analysis/a5_learned_behaviour.py --run_dir runs/gemma_v4 --compare_run runs/abl_random \
      --cities tokyo sydney moscow --probe_src istanbul new_york
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (SRC, TGT, base_args, load_cfg, taxonomy, save_table, setup_mpl, save_fig, pretty, short_act)

EXPLAIN_R = {"hist": "(1+recency-weighted visits)^w  preferential return",
             "prefix": "(1+visits in current trip)^w", "is_last": "exp(w) for staying at current place",
             "ptrans": "(1+own transitions last->v)^w  personal habit",
             "tod": "(1+visits at this time of day)^w  daily rhythm",
             "dow": "(1+visits on this day type)^w  weekly rhythm",
             "rec": "exp(w * e^{-days/14})  recency boost", "geo": "(1+km)^-w  distance decay",
             "lptr": "latent pointer (transformer) weight"}
EXPLAIN_E = {"hier": "[p(primary)p(secondary)]^w  category model", "geo": "(1+km)^-w  gravity",
             "anchor": "(1+km to user's anchors)^-w", "covis": "city co-visitation (needs city log)",
             "pop": "city popularity (needs city log)", "hour": "city hour profile (needs city log)"}


def load_model(cfg, run_dir, city, device, fsq, enc, logger, random_backbone=False):
    from gemma.data import build_city_data
    from gemma.downstream import load_backbone, split_ids, ALL_SPLITS
    from gemma.realize import CityIndex, new_head
    zs = torch.load(os.path.join(run_dir, "zs_hrem.pt"), map_location="cpu", weights_only=False)
    cfg.retriever = "hrem"; cfg.random_backbone = random_backbone
    C = build_city_data(cfg, city, ALL_SPLITS, fsq, enc, logger, filter_users=False)
    ids = split_ids(C)
    bb = load_backbone(cfg, run_dir, device, zs.get("backbone"))
    ci = CityIndex(C, observed=ids["train"], stat=None, cfg=cfg, device=device)
    head = new_head(bb, ci, cfg, device); head.load_state_dict(zs["head"], strict=False); head.eval()
    return bb, head, ci, ids


@torch.no_grad()
def collect(bb, head, ci, qs, cfg, bs=64, max_q=4000, rng=None):
    from gemma.realize import _forward, _amp
    if rng is not None and len(qs) > max_q:
        qs = [qs[i] for i in rng.choice(len(qs), max_q, replace=False)]
    out = {k: [] for k in ("wr", "we", "pi", "new", "act", "S", "z", "rank")}
    for s in range(0, len(qs), bs):
        o = _forward(bb, head, ci, qs[s:s + bs], False, _amp(cfg, ci.device), 0)
        m = o["m"]; tgt = m["tgt_pi"]; ar = torch.arange(len(tgt), device=tgt.device)
        a_t = m["tgt_act"].clamp(max=o["gate"].size(1) - 1)
        out["wr"].append(o["wr"].float().cpu()); out["we"].append(o["we"].float().cpu())
        out["pi"].append(torch.sigmoid(o["gate"][ar, a_t].float()).cpu())
        out["new"].append((~o["visited"][ar, tgt]).float().cpu())
        out["act"].append(a_t.cpu()); out["S"].append(m["u_S"].float().cpu())
        out["z"].append(o["z1"].float().cpu())
        lg = o["logits"].float(); out["rank"].append((lg > lg.gather(1, tgt[:, None])).sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def probe(Xtr, ytr, Xte, yte, n_cls, steps=300, wd=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); Xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long); yte = torch.tensor(yte, dtype=torch.long)
    lin = torch.nn.Linear(Xtr.size(1), n_cls)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(lin(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        return float((lin(Xte).argmax(1) == yte).float().mean()), float((lin(Xtr).argmax(1) == ytr).float().mean())


def main():
    ap = base_args(__doc__, ["tokyo", "sydney", "moscow"])
    ap.add_argument("--compare_run", default="runs/abl_random", help="backbone without JEPA pretraining")
    ap.add_argument("--probe_src", nargs="+", default=["istanbul", "new_york"])
    ap.add_argument("--max_q", type=int, default=4000)
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    cfg = load_cfg(a); cfg.allow_cpu = a.allow_cpu
    from gemma.fsq import FSQTaxonomy
    from gemma.poi_encoder import POIEncoder
    from gemma.utils import check_device, get_logger
    from gemma.realize import sthgcn_filter
    from gemma.hrem import RET, EXP
    logger = get_logger("gemma.a5")
    device = check_device(cfg, logger)
    fsq = FSQTaxonomy(cfg.fsq_csv); enc = POIEncoder(cfg, logger); cfg.poi_emb_dim = enc.dim
    names = {v: k for k, v in fsq.act2id.items()}
    plt = setup_mpl()

    def city_data(run_dir, c, rb):
        bb, head, ci, ids = load_model(cfg, run_dir, c, device, fsq, enc, logger, rb)
        q = sthgcn_filter(ci, ci.queries(ids["test"], False), ids["train"])
        return collect(bb, head, ci, q, cfg, max_q=a.max_q, rng=rng)

    R = {c: city_data(a.run_dir, c, False) for c in a.cities}
    # (a) learned weights
    WR = np.concatenate([R[c]["wr"] for c in R]); WE = np.concatenate([R[c]["we"] for c in R])
    rows = []
    for j, k in enumerate(RET):
        rows.append({"channel": "return", "evidence": k, "mean w": WR[:, j].mean(),
                     "p10": np.percentile(WR[:, j], 10), "p90": np.percentile(WR[:, j], 90),
                     "meaning": EXPLAIN_R.get(k, "")})
    for j, k in enumerate(EXP):
        rows.append({"channel": "explore", "evidence": k, "mean w": WE[:, j].mean(),
                     "p10": np.percentile(WE[:, j], 10), "p90": np.percentile(WE[:, j], 90),
                     "meaning": EXPLAIN_E.get(k, "")})
    save_table(pd.DataFrame(rows), a.out, "A5_learned_weights",
               "Evidence weights learned on the source cities, evaluated on held-out-city test queries "
               "(mean and 10th--90th percentile over queries). Weights are exponents of a product of experts.")
    # (b) gate calibration
    pi = np.concatenate([R[c]["pi"] for c in R]); new = np.concatenate([R[c]["new"] for c in R])
    act = np.concatenate([R[c]["act"] for c in R]); S = np.concatenate([R[c]["S"] for c in R])
    rows = []
    for k in np.unique(act):
        sel = act == k
        if sel.sum() >= 30:
            rows.append({"activity": names.get(int(k), str(k)), "n": int(sel.sum()),
                         "predicted P(explore)": float(pi[sel].mean()), "observed share new": float(new[sel].mean())})
    G = pd.DataFrame(rows).sort_values("observed share new")
    save_table(G, a.out, "A5_gate_calibration",
               "Activity-conditioned return/explore gate on held-out cities: predicted probability of "
               "visiting a new place vs observed share.")
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    ax.scatter(G["observed share new"], G["predicted P(explore)"], s=12 + G["n"] / G["n"].max() * 60, color="#0072B2")
    for _, r in G.iterrows():
        ax.annotate(r["activity"][:14], (r["observed share new"], r["predicted P(explore)"]), fontsize=5.5)
    ax.plot([0, 1], [0, 1], "k:", lw=0.6); ax.set_xlabel("observed share of new places"); ax.set_ylabel("GEMMA P(explore)")
    save_fig(fig, a.out, "A5a_gate_by_activity"); plt.close(fig)
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(pi, bins) - 1, 0, 9)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    xs = [pi[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
    ys = [new[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
    ax.plot(xs, ys, "o-", color="#0072B2"); ax.plot([0, 1], [0, 1], "k:", lw=0.6)
    ax.set_xlabel("predicted P(explore)"); ax.set_ylabel("observed share new")
    save_fig(fig, a.out, "A5b_reliability"); plt.close(fig)
    sb = np.unique(np.clip(np.round(np.log2(S + 1)), 0, 10))
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    xs, yp, yo = [], [], []
    for b in sb:
        sel = np.clip(np.round(np.log2(S + 1)), 0, 10) == b
        if sel.sum() >= 20:
            xs.append(2 ** b); yp.append(pi[sel].mean()); yo.append(new[sel].mean())
    ax.plot(xs, yo, "s-", color="#999999", label="observed"); ax.plot(xs, yp, "o-", color="#0072B2", label="GEMMA")
    ax.set_xscale("log"); ax.set_xlabel("distinct places in history $S$"); ax.set_ylabel("P(next place is new)")
    ax.legend(frameon=False)
    save_fig(fig, a.out, "A5c_epr_learned"); plt.close(fig)
    # (c) representation probes: JEPA backbone vs backbone without JEPA pretraining
    rows = []
    variants = [("JEPA-pretrained", a.run_dir, False)]
    if a.compare_run and os.path.exists(os.path.join(a.compare_run, "zs_hrem.pt")):
        variants.append(("no JEPA", a.compare_run, True))
    pca_done = False
    for lab, rd, rb in variants:
        src = {c: city_data(rd, c, rb) for c in a.probe_src}
        tgt = R if rd == a.run_dir else {c: city_data(rd, c, rb) for c in a.cities}
        Xs = np.concatenate([src[c]["z"] for c in src]); ya = np.concatenate([src[c]["act"] for c in src])
        yn = np.concatenate([src[c]["new"] for c in src]).astype(int)
        n_cls = int(max(ya.max(), max(tgt[c]["act"].max() for c in tgt))) + 1
        for c in tgt:
            acc_a, tr_a = probe(Xs, ya, tgt[c]["z"], tgt[c]["act"], n_cls)
            acc_n, _ = probe(Xs, yn, tgt[c]["z"], tgt[c]["new"].astype(int), 2)
            maj = float(np.bincount(tgt[c]["act"]).max() / len(tgt[c]["act"]))
            rows.append({"backbone": lab, "target city": pretty(c), "next-activity probe": acc_a,
                         "majority": maj, "revisit/explore probe": acc_n,
                         "revisit majority": float(max(tgt[c]["new"].mean(), 1 - tgt[c]["new"].mean()))})
        if not pca_done:
            Z = np.concatenate([tgt[c]["z"] for c in tgt]); Z = Z - Z.mean(0)
            ev = np.linalg.svd(Z, compute_uv=False) ** 2; ev = ev / ev.sum()
            U, Sg, Vt = np.linalg.svd(Z, full_matrices=False)
            fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
            axes[0].plot(np.arange(1, min(51, len(ev) + 1)), np.cumsum(ev)[:50], color="#0072B2")
            axes[0].set_xlabel("principal directions of z"); axes[0].set_ylabel("variance explained")
            Y = Z @ Vt[:2].T; acts_all = np.concatenate([tgt[c]["act"] for c in tgt])
            from common import ACT_COLORS
            for k in np.unique(acts_all)[:12]:
                sel = acts_all == k
                axes[1].scatter(Y[sel, 0], Y[sel, 1], s=2, alpha=0.4, color=ACT_COLORS[int(k) % len(ACT_COLORS)],
                                label=short_act(names.get(int(k), str(k))))
            axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend(fontsize=5, markerscale=4, frameon=False)
            save_fig(fig, a.out, "A5d_latent_pca"); plt.close(fig); pca_done = True
    save_table(pd.DataFrame(rows), a.out, "A5_probes",
               "Linear probes on the frozen backbone latent, trained on source cities and tested on held-out "
               "cities (no target-city training).")


if __name__ == "__main__":
    main()
