"""paper/figures.py -- every data figure of the GEMMA paper from paper/data/results.json.

  python paper/collect.py --logs logs --analysis analysis_out/tables   # refresh the numbers
  python paper/figures.py                    # all figures -> paper/out/figs/*.pdf|png
  python paper/figures.py fig3 fig4          # a subset

fig1_teaser        identifier shift / universal law / headline accuracy           (figure*, top of p.1-2)
fig3_main_forest   per-city Delta Acc@1 vs STHGCN with 95% CI (unseen + source)    (figure*)
fig4_llm           zero-shot GEMMA vs LLM agents, per unseen city                  (column)
fig5_anatomy       accuracy = returns + explorations; explore ladder; signal value (figure*)
fig6_weights       learned evidence weights and gate; stability under fine-tuning (figure*)
fig7_ablation      learned vs hand-set evidence weights; backbone objectives       (column)
fig8_finetune      validation Acc@1 during fine-tuning (unseen / source)            (column)
fig9_evidence_ablation  leave-one-evidence-out + gate ablations (needs run_paper_ablate.sh)  (column)
fig10_history      accuracy vs. visible history length, cold start (needs run_paper_ablate.sh) (column)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_api import Results, ztest
from style import (BASE, COL, EXP, GRID, INK, INK2, LLM, LLM_LIGHT, MUTED, N_TGT, OURS, OURS_LIGHT, RET, SOTA, SRC,
                   TEXT, TGT, grid_x, grid_y, panel, pretty, save, setup)

R = None
WORD = {6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}


def _val(rows, key):
    for r in rows:
        if key in r:
            return r
    return None


# =============================================================================== Figure 1
def fig1_teaser():
    fig = plt.figure(figsize=(TEXT, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=.62)

    # (a) identifier shift: cross-city divergence of each representation
    ax = fig.add_subplot(gs[0])
    lad = {r["representation"]: r for r in R.table("A2_abstraction_ladder")}
    first = next(iter(lad.values())) if lad else {}
    key = next((k for k in ("cross-city JS (half vs half)", "cross-city JS", "mean cross-city JS (bits)") if k in first), None)
    levels = [("POI (venue id)", "place ID"), ("category (leaf)", "leaf category"), ("activity x hour", "activity × hour"),
              ("primary category", "primary category"), ("activity transition", "activity transition"), ("activity", "activity")]
    vals, labs = [], []
    for k, lab in levels:
        # place ids are city-local in Massive-STEPS: disjoint supports => JS divergence = 1 bit exactly
        v = 1.0 if k == "POI (venue id)" else (lad[k][key] if k in lad else np.nan)
        vals.append(v)
        labs.append(lab)
    y = np.arange(len(vals))[::-1]
    ax.barh(y[1:], vals[1:], color=INK2, height=.62)
    ax.barh(y[0], vals[0], color="white", edgecolor=BASE, hatch="////", height=.62, linewidth=.6)
    for yi, v in zip(y, vals):
        ax.text(v + .02, yi, f"{v:.2f}", va="center", fontsize=6.5, color=INK)
    ax.text(.5, y[0], "disjoint IDs", ha="center", va="center", fontsize=6.3, color=INK2,
            bbox=dict(fc="white", ec="none", pad=.6))
    ax.set_yticks(y, labs)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, .5, 1])
    ax.set_xlabel("cross-city divergence (JS, bits)")
    grid_x(ax)
    panel(ax, "a", "Identifier shift", x=-.62)

    # (b) universal law: novelty exponent per city vs the learned gate coefficient
    ax = fig.add_subplot(gs[1])
    A1 = R.table("A1_mobility_laws")
    g = sorted(((r["gamma (Pnew~S^-g)"], r["city"], r["role"]) for r in A1))
    yy = np.arange(len(g))
    W = R.source_weights()
    gs_ = np.array([-w["gate"]["logS"] for w in W])
    lo, hi = gs_.mean() - max(gs_.std(ddof=1), .01), gs_.mean() + max(gs_.std(ddof=1), .01)
    ax.axvspan(lo, hi, color=OURS, alpha=.22, lw=0)
    ax.axvline(gs_.mean(), color=OURS, lw=1.2)
    for yi, (gv, c, role) in zip(yy, g):
        ax.plot([0.1, gv], [yi, yi], color=GRID, lw=.6, zorder=1)
        ax.scatter(gv, yi, s=16, marker="s" if role == "source" else "o",
                   facecolor=INK2 if role == "source" else "white", edgecolor=INK2, lw=.8, zorder=3)
    ax.set_yticks(yy, [c for _, c, _ in g], fontsize=6.2)
    ax.set_xlim(.1, .55)
    ax.set_xlabel(r"novelty exponent $\gamma$  ($P_{\rm new}\propto S^{-\gamma}$)")
    ax.text(gs_.mean() + .012, len(g) - .6, "GEMMA's\nlearned $-g_S$", color=OURS, fontsize=6.3, va="top", ha="left")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc=INK2, mec=INK2, ms=3.5, label="source city"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.5, label="unseen city")],
              loc="lower right", fontsize=6.2, handletextpad=.2, borderaxespad=0)
    panel(ax, "b", "A universal decision law", x=-.42)

    # (c) headline accuracy on the unseen cities.  solid = trained on the target city, light = zero-shot
    ax = fig.add_subplot(gs[2])
    zs, ft = R.agg("zs", TGT), R.agg("ft", TGT)
    bars = [("GETNext", R.sup_agg("GETNext", TGT)["pooled"], 0, BASE, BASE),
            ("STHGCN", R.sup_agg("STHGCN", TGT)["pooled"], 0, SOTA, SOTA),
            ("GEMMA\nZS", zs["pooled"][0], zs["pooled"][1], OURS_LIGHT, OURS),
            ("GEMMA\nFT", ft["pooled"][0], ft["pooled"][1], OURS, OURS)]
    llmb = np.mean([R.llm_best(c)[0] for c in TGT])
    gl = R.llm_proto()
    gl_m = np.mean([gl[c][0] for c in TGT])
    gl_sd = np.std([np.mean([r["rand_acc1"] for r in R.R["llm"] if r["seed"] == s]) for s in
                    sorted({r["seed"] for r in R.R["llm"]})], ddof=1)
    bars2 = [("best LLM\nagent", llmb, 0, LLM_LIGHT, LLM), ("GEMMA\nZS", gl_m, gl_sd, OURS_LIGHT, OURS)]
    # horizontal bars: two protocol groups, top to bottom
    y1 = np.arange(len(bars))[::-1] + len(bars2) + 1.3
    y2 = np.arange(len(bars2))[::-1]
    for ys_, bs in ((y1, bars), (y2, bars2)):
        for yi, (lab, v, sd, fc, ec) in zip(ys_, bs):
            ax.barh(yi, v, height=.68, color=fc, edgecolor=ec, lw=.9)
            if sd:
                ax.errorbar(v, yi, xerr=sd, color=INK, lw=.7, capsize=1.5)
            ax.text(v + (sd or 0) + .008, yi, f"{v:.3f}", ha="left", va="center", fontsize=6.4)
    ax.set_yticks(list(y1) + list(y2), [b[0].replace("\n", " ") for b in bars] + [b[0].replace("\n", " ") for b in bars2],
                  fontsize=6.5)
    ax.axhline(len(bars2) + .15, color=GRID, lw=.8)
    ax.text(.585, y1[0] + .55, f"supervised protocol (pooled, {sum(R.n[c] for c in TGT) / 1000:.0f}k queries)", ha="right", va="bottom", fontsize=6.1, color=INK2)
    ax.text(.585, y2[0] + .55, "LLM-agent protocol (truth + 100 random)", ha="right", va="bottom", fontsize=6.1, color=INK2)
    ax.annotate("", xy=(gl_m, y2[1] + .36), xytext=(llmb + .01, y2[0] - .02),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=.9, connectionstyle="arc3,rad=.25", mutation_scale=7))
    ax.text(gl_m + .095, y2[1] + .02, f"×{gl_m / llmb:.1f}", ha="left", va="center", fontsize=8, color=OURS, fontweight="bold")
    ax.set_xlim(0, .6)
    ax.set_ylim(-.6, y1[0] + 1.25)
    ax.set_xlabel(f"Acc@1 on the {N_TGT} unseen cities")
    grid_x(ax)
    ax.legend(handles=[Patch(facecolor=SOTA, edgecolor=SOTA, label="target-city training"),
                       Patch(facecolor=OURS_LIGHT, edgecolor=OURS, label="zero-shot")],
              loc="center", bbox_to_anchor=(.6, .375), ncol=2, fontsize=6.1, handlelength=1, handletextpad=.4, columnspacing=.9)
    panel(ax, "c", "One model for unseen cities", x=-.3)
    save(fig, "fig1_teaser")


# =============================================================================== Figure 3
def fig3_main_forest():
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.55), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=.62))
    # (a) unseen cities: zero-shot and fine-tuned
    ax = axes[0]
    XL = (-7, 7)
    zs, ft = R.city("zs"), R.city("ft")
    order = sorted(TGT, key=lambda c: R.n[c])
    ys = np.arange(len(order)) + 1.2
    for yi, c in zip(ys, order):
        ref = R.sup("STHGCN", c)
        for (m, sd, k), setting, dy, mfc in ((zs[c], "zs", .17, "white"), (ft[c], "ft", -.17, OURS)):
            d, e = 100 * (m - ref), 100 * R.ci(setting, c)
            lo = max(d - e, XL[0])
            ax.plot([lo, d + e], [yi + dy] * 2, color=OURS, lw=1.0, solid_capstyle="round")
            if d - e < XL[0]:     # CI truncated at the axis: mark with an arrow head
                ax.annotate("", xy=(XL[0], yi + dy), xytext=(XL[0] + .9, yi + dy),
                            arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.0, mutation_scale=6))
            ax.scatter(d, yi + dy, s=18, marker="o", facecolor=mfc, edgecolor=OURS, lw=1.0, zorder=3)
    # pooled row
    for setting, dy, mfc in (("zs", .17, "white"), ("ft", -.17, OURS)):
        a = R.agg(setting, TGT)["pooled"]
        ref = R.sup_agg("STHGCN", TGT)["pooled"]
        n = sum(R.n[c] for c in TGT)
        e = 100 * 1.96 * np.sqrt(a[0] * (1 - a[0]) / n + a[1] ** 2 / 3)
        ax.plot([100 * (a[0] - ref) - e, 100 * (a[0] - ref) + e], [0 + dy] * 2, color=OURS, lw=1.0)
        ax.scatter(100 * (a[0] - ref), dy, s=26, marker="D", facecolor=mfc, edgecolor=OURS, lw=1.0, zorder=3)
    ax.axhline(.65, color=GRID, lw=.8)
    ax.axvline(0, color=SOTA, lw=.9)
    ax.set_yticks([0] + list(ys), [f"all {N_TGT} (pooled)"] + [f"{pretty(c)}  ({R.n[c]:,})" for c in order], fontsize=6.6)
    ax.set_xlabel("Δ Acc@1 vs. STHGCN (points)  ·  95% CI")
    ax.set_xlim(*XL)
    grid_x(ax)
    ax.text(.3, len(order) + 1.05, "STHGCN\n(trained in each city)", fontsize=6.2, color=SOTA, va="bottom", ha="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="-", color=OURS, mfc="white", ms=4, lw=1, label="GEMMA zero-shot (no target data)"),
                       Line2D([], [], marker="o", ls="-", color=OURS, mfc=OURS, ms=4, lw=1, label="GEMMA fine-tuned (train split only)")],
              loc="upper left", bbox_to_anchor=(0, -.2), ncol=2, fontsize=6.3, borderaxespad=.1, handletextpad=.4)
    panel(ax, "a", f"{WORD[N_TGT]} unseen cities", x=-.42)

    # (b) source cities
    ax = axes[1]
    sc = R.city("src")
    order = sorted(SRC, key=lambda c: R.n[c])
    ys = np.arange(len(order)) + 1.2
    for yi, c in zip(ys, order):
        ref = R.sup("STHGCN", c)
        m = sc[c][0]
        d, e = 100 * (m - ref), 100 * R.ci("src", c)
        ax.plot([d - e, d + e], [yi] * 2, color=OURS, lw=1.0)
        ax.scatter(d, yi, s=18, facecolor=OURS, edgecolor=OURS, zorder=3)
    a = R.agg("src", SRC)["pooled"]
    ref = R.sup_agg("STHGCN", SRC)["pooled"]
    n = sum(R.n[c] for c in SRC)
    e = 100 * 1.96 * np.sqrt(a[0] * (1 - a[0]) / n + a[1] ** 2 / 3)
    ax.plot([100 * (a[0] - ref) - e, 100 * (a[0] - ref) + e], [0, 0], color=OURS, lw=1.0)
    ax.scatter(100 * (a[0] - ref), 0, s=26, marker="D", facecolor=OURS, edgecolor=OURS, zorder=3)
    ax.axhline(.65, color=GRID, lw=.8)
    ax.axvline(0, color=SOTA, lw=.9)
    ax.set_yticks([0] + list(ys), ["all 6 (pooled)"] + [f"{pretty(c)}  ({R.n[c]:,})" for c in order], fontsize=6.6)
    ax.set_xlabel("Δ Acc@1 vs. STHGCN (points)  ·  95% CI")
    ax.set_xlim(-6, 3)
    grid_x(ax)
    panel(ax, "b", "Six source cities (fine-tuned)", x=-.62)
    save(fig, "fig3_main_forest")


# =============================================================================== Figure 4
SHOW_NEAR = False      # True: add GEMMA with the 100 NEAREST places as distractors (a harder set than the LLMs saw)


def fig4_llm():
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    gp = R.llm_proto("rand_acc1")
    gn = R.llm_proto("near_acc1")
    order = sorted(TGT, key=lambda c: gp[c][0])
    ys = np.arange(len(order))
    for yi, c in zip(ys, order):
        best, who = R.llm_best(c)
        ax.plot([best, gp[c][0]], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(best, yi, s=22, color=LLM, zorder=3)
        ax.errorbar(gp[c][0], yi, xerr=gp[c][1], fmt="o", ms=4.5, color=OURS, mec=OURS, elinewidth=.8, zorder=4)
        ax.text(gp[c][0] + gp[c][1] + .018, yi, f"×{gp[c][0] / best:.1f}", va="center", fontsize=6.4, color=OURS)
        if SHOW_NEAR:
            ax.scatter(gn[c][0], yi, s=16, marker="D", facecolor="white", edgecolor=OURS, lw=.9, zorder=3)
    ax.set_yticks(ys, [pretty(c) for c in order])
    ax.set_xlim(0, .7)
    ax.set_xlabel("Acc@1  (ground truth + 100 random places)")
    grid_x(ax)
    h = [Line2D([], [], marker="o", ls="", color=LLM, ms=4.2, label="best of 15 published LLM agents, per city"),
         Line2D([], [], marker="o", ls="", color=OURS, ms=4.5, label="GEMMA zero-shot (mean ± sd, 3 seeds)")]
    if SHOW_NEAR:
        h.append(Line2D([], [], marker="D", ls="", mfc="white", mec=OURS, ms=3.8, label="GEMMA, 100 nearest candidates (harder)"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(.42, -.2), fontsize=6.3, ncol=1, handletextpad=.3)
    save(fig, "fig4_llm")


# =============================================================================== Figure 5
def fig5_anatomy():
    fig = plt.figure(figsize=(TEXT, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.05], wspace=.78)

    # (a) Acc@1 = rho*A_R + (1-rho)*A_E
    ax = fig.add_subplot(gs[0])
    rows = R.rows("zs")
    order = sorted(TGT, key=lambda c: R.city("zs")[c][0])
    ys = np.arange(len(order))
    for yi, c in zip(ys, order):
        rr = [r for r in rows if r["city"] == c]
        rho = np.mean([r["rho"] for r in rr])
        ar = np.mean([r["ret_acc1"] for r in rr])
        ae = np.mean([r["exp_acc1"] for r in rr])
        ax.barh(yi, rho * ar, color=RET, height=.62)
        ax.barh(yi, (1 - rho) * ae, left=rho * ar, color=EXP, height=.62)
        ax.plot([R.sup("STHGCN", c)] * 2, [yi - .38, yi + .38], color=SOTA, lw=1.4)
    ax.set_yticks(ys, [pretty(c) for c in order])
    ax.set_xlabel("zero-shot Acc@1")
    ax.set_xlim(0, .5)
    grid_x(ax)
    ax.legend(handles=[Patch(color=RET, label=r"returns $\rho A_R$"), Patch(color=EXP, label=r"explore $(1-\rho)A_E$"),
                       Line2D([], [], color=SOTA, lw=1.4, label="STHGCN")], loc="upper center", bbox_to_anchor=(.38, -.2),
              ncol=3, fontsize=6.0, handlelength=.9, handletextpad=.3, columnspacing=.7)
    panel(ax, "a", "Accuracy is made of returns", x=-.5)

    # (b) explore ranking: oracles vs GEMMA (conditional on the target being new)
    ax = fig.add_subplot(gs[1])
    O = [r for r in R.table("A8_explore_oracles") if r["city"] != "mean"]
    tg = O
    def m(col):
        return float(np.mean([r[col] for r in tg]))
    entries = [("nearest", "nearest"), ("popularity", "popularity"), ("gravity", "gravity (pop/dist)"),
               ("nearest | activity*", "nearest | activity*"), ("gravity | activity*", "gravity | activity*"),
               ("popularity | category*", "popularity | category*"), ("nearest | category*", "nearest | category*")]
    vals = [(lab, m(col + " @1"), m(col + " @10"), "*" in lab) for lab, col in entries]
    zr = R.rows("zs")
    fr = R.rows("ft")
    vals.append(("GEMMA zero-shot", np.mean([r["cexp1"] for r in zr]), np.mean([r["cexp10"] for r in zr]), False))
    vals.append(("GEMMA fine-tuned", np.mean([r["cexp1"] for r in fr]), np.mean([r["cexp10"] for r in fr]), False))
    vals.sort(key=lambda t: t[2])
    ys = np.arange(len(vals))
    for yi, (lab, a1, a10, oracle) in zip(ys, vals):
        is_g = lab.startswith("GEMMA")
        col = OURS if is_g else (BASE if oracle else INK2)
        ax.barh(yi, a10, color=col, alpha=.38 if not is_g else .45, height=.66,
                hatch="////" if oracle else None, edgecolor="white" if oracle else None, lw=0)
        ax.barh(yi, a1, color=col, height=.66)
    ax.set_yticks(ys, [v[0] for v in vals], fontsize=6.3)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("GEMMA"):
            t.set_color(OURS); t.set_fontweight("bold")
    ax.set_xlabel("explore Acc@1 (dark), Acc@10 (light)")
    ax.set_xlim(0, .8)
    grid_x(ax)
    panel(ax, "b", "Exploration is the hard part", x=-.75)

    # (c) information value of single signals: macro over the evaluated cities, from the per-city A4 log
    ax = fig.add_subplot(gs[2])
    meta = {r["signal"]: r for r in R.table("A4_information_value")}
    pc = [r for r in R.table("A4_per_city") if r["city"] in TGT]
    iv = []
    for sig, r in meta.items():
        vals = [p[sig] for p in pc if sig in p]
        a1 = float(np.mean(vals)) if len(vals) == len(TGT) else r["macro Acc@1"]
        iv.append({**r, "macro Acc@1": a1})
    iv = sorted(iv, key=lambda r: r["macro Acc@1"])
    ys = np.arange(len(iv))
    for yi, r in zip(ys, iv):
        personal = r["knowledge"] == "user history"
        used = str(r["used by GEMMA zero-shot"]).startswith("yes")
        ax.barh(yi, r["macro Acc@1"], color=(RET if personal else INK2), height=.66,
                hatch=None if used else "////", edgecolor="white" if not used else None)
        ax.text(r["macro Acc@1"] + .005, yi, f"{r['macro Acc@1']:.3f}", va="center", fontsize=6.0)
    ax.set_yticks(ys, [r["signal"].replace("activity prior x distance", "activity × distance") for r in iv], fontsize=6.3)
    ax.set_xlabel("Acc@1 of the signal alone")
    ax.set_xlim(0, .29)
    grid_x(ax)
    ax.legend(handles=[Patch(color=RET, label="user's own history"), Patch(color=INK2, label="geometry / city log"),
                       Patch(facecolor=INK2, hatch="////", edgecolor="white", label="needs target-city log")],
              loc="lower right", fontsize=6.0, handlelength=1.0, handletextpad=.4)
    panel(ax, "c", "Where the signal is", x=-.62)
    save(fig, "fig5_anatomy")


# =============================================================================== Figure 6
RET_NAMES = {"hist": "frequency", "prefix": "in this trip", "is_last": "is last place", "ptrans": "last→next",
             "tod": "hour of day", "dow": "weekday", "rec": "recency", "geo": "distance", "lptr": "latent pointer"}
EXP_NAMES = {"hier": "activity path", "geo": "distance", "anchor": "anchor distance", "covis": "co-visit†",
             "pop": "popularity†", "hour": "city hour profile†"}
GATE_NAMES = {"nov": r"$g_\nu$ novelty rate", "logS": r"$g_S$ log #places", "pref": r"$g_p$ new prefix"}


def _weight_panel(ax, names, part, title, letter, color):
    src = R.source_weights()
    ftv = R.finetuned_weights("V")
    fts = R.finetuned_weights("S")
    keys = list(names)
    x = np.arange(len(keys))
    sv = np.array([[w[part][k] for k in keys] for w in src])
    ax.bar(x, sv.mean(0), yerr=sv.std(0, ddof=1) if len(sv) > 1 else None, color=color, width=.62,
           error_kw=dict(lw=.7, capsize=1.5, ecolor=INK))
    rng = np.random.default_rng(0)
    for d, mk, fc in ((ftv, "o", "white"), (fts, "s", "white")):
        for (s, c), w in d.items():
            vals = [w[part][k] for k in keys]
            ax.scatter(x + rng.uniform(-.2, .2, len(x)), vals, s=6, marker=mk, facecolor=fc, edgecolor=INK2, lw=.45, zorder=3)
    ax.axhline(0, color=INK2, lw=.6)
    ax.set_xticks(x, [names[k] for k in keys], rotation=40, ha="right", fontsize=6.4)
    grid_y(ax)
    panel(ax, letter, title, x=-.12)


def fig6_weights():
    fig = plt.figure(figsize=(TEXT, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.45, .75], wspace=.32)
    ax = fig.add_subplot(gs[0])
    _weight_panel(ax, RET_NAMES, "ret", "Return evidence weights", "a", RET)
    ax.set_ylabel("learned weight (at z = 0)")
    ax.legend(handles=[Patch(color=RET, label="trained on 6 source cities (3 seeds)"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3, label="after fine-tuning, unseen city"),
                       Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, ms=3, label="after fine-tuning, source city")],
              loc="lower right", fontsize=6.0, handletextpad=.3)
    ax = fig.add_subplot(gs[1])
    _weight_panel(ax, EXP_NAMES, "exp", "Explore evidence weights", "b", EXP)
    ax.text(.99, .97, "† needs the target-city log:\n   inactive in zero-shot", transform=ax.transAxes, ha="right",
            va="top", fontsize=5.9, color=INK2)
    ax = fig.add_subplot(gs[2])
    _weight_panel(ax, GATE_NAMES, "gate", "Gate", "c", OURS)
    A1 = R.table("A1_mobility_laws")
    gam = np.array([r["gamma (Pnew~S^-g)"] for r in A1])
    ax.axhspan(-np.percentile(gam, 75), -np.percentile(gam, 25), color=MUTED, alpha=.25, lw=0)
    ax.axhline(-gam.mean(), color=MUTED, lw=.9, ls="--")
    ax.set_ylim(-.52, None)
    ax.text(2.45, -.5, rf"$-\bar\gamma$, {len(gam)} cities", fontsize=5.9, color=INK2, ha="right", va="bottom")
    save(fig, "fig6_weights")


# =============================================================================== Figure 7
def fig7_ablation():
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.25), gridspec_kw=dict(height_ratios=[1, 1.15], hspace=.75))
    # (a) learned vs hand-set evidence weights (zero-shot, same backbone)
    ax = axes[0]
    rep = R.report("final_R_zero")
    order = [c for c in TGT if c in rep.get("hand", {})]
    order = sorted(order, key=lambda c: R.n[c])
    d = [100 * (rep["v6"][c]["acc1"] - rep["hand"][c]["acc1"]) for c in order]
    sig = [rep["hand"][c]["dsig"] for c in order]
    x = np.arange(len(order))
    ax.bar(x, d, color=[OURS if s_ else OURS_LIGHT for s_ in sig], width=.66)
    mac = 100 * (np.mean([rep["v6"][c]["acc1"] for c in order]) - np.mean([rep["hand"][c]["acc1"] for c in order]))
    ax.axhline(mac, color=INK, lw=.8, ls="--")
    ax.set_xticks(x, [pretty(c) for c in order], rotation=35, ha="right", fontsize=6.3)
    ax.set_ylabel("Δ Acc@1 (points)")
    grid_y(ax)
    ax.legend(handles=[Patch(color=OURS, label="paired 95% CI excludes 0"), Patch(color=OURS_LIGHT, label="not significant"),
                       Line2D([], [], color=INK, ls="--", lw=.8, label=f"macro +{mac:.1f}")],
              loc="upper left", fontsize=6.0, ncol=3, columnspacing=.8, handlelength=1, borderaxespad=0, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, 6.2)
    panel(ax, "a", "Learned vs. hand-set evidence weights (zero-shot)", x=-.2)
    # (b) representation-learning objectives
    ax = axes[1]
    rep = R.report("final_R_ablation")
    names = [n for n in rep if not n.startswith("_") and n != "full"]
    lab = {"noWM": "no world model", "noJEPA": "no JEPA loss", "noPtr": "no latent pointer", "noLat": "no latent match",
           "noDT": "no Δt features", "addMotif": "+ motif prior", "flat": "flat taxonomy", "H1": "1-level hierarchy",
           "randomInit": "random init (no pretraining)"}
    rows = []
    for nme in names:
        dd = [100 * (rep[nme][c]["acc1"] - rep["full"][c]["acc1"]) for c in TGT if c in rep[nme]]
        rows.append((lab.get(nme, nme), np.mean(dd), dd))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    for yi, (l, m, dd) in zip(ys, rows):
        ax.scatter(dd, [yi] * len(dd), s=5, color=MUTED, zorder=2)
        ax.scatter(m, yi, s=22, marker="D", color=INK, zorder=3)
    ax.axvline(0, color=SOTA, lw=.8)
    ax.axvspan(-.5, .5, color=GRID, alpha=.6, lw=0)
    ax.text(.49, len(rows) - .45, "±0.5 pt", ha="right", va="bottom", fontsize=5.9, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=6.3)
    ax.set_xlabel("Δ zero-shot Acc@1 vs. full backbone (points)")
    grid_x(ax)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=3.5, label=f"macro ({N_TGT} cities)"),
                       Line2D([], [], marker="o", ls="", color=MUTED, ms=2.5, label="per city")],
              loc="lower right", fontsize=6.0, handletextpad=.2)
    panel(ax, "b", "Backbone pretraining objectives", x=-.62)
    save(fig, "fig7_ablation")


# =============================================================================== Figure 8
def fig8_finetune():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.8), gridspec_kw=dict(wspace=.42, width_ratios=[1.35, 1]))
    for ax, run, cities, letter, title in ((axes[0], "V", TGT, "a", "Unseen cities"), (axes[1], "S", SRC, "b", "Source cities")):
        cur = R.val_curves(run)
        curves = {}
        for c in cities:
            tr = [v for (s_, cc), v in cur.items() if cc == c]
            if not tr:
                continue
            L = max(len(t) for t in tr)
            arr = np.full((len(tr), L), np.nan)
            for i, t in enumerate(tr):        # early stopping: shorter runs keep their last (restored-best) value
                vals = [a for _, a in t]
                arr[i, :len(vals)] = vals
            m = np.nanmean(arr, 0)
            curves[c] = 100 * (m - m[0])
            ax.plot(np.arange(L), curves[c], color=BASE, lw=.8)
        L = max(len(g) for g in curves.values())
        M = np.full((len(curves), L), np.nan)
        for i, g in enumerate(curves.values()):
            M[i, :len(g)] = g
        keep = np.sum(~np.isnan(M), 0) >= max(2, len(curves) // 2)
        med = np.nanmedian(M, 0)
        ax.plot(np.arange(L)[keep], med[keep], color=OURS, lw=1.6, marker="o", ms=2.5)
        ax.axhline(0, color=INK2, lw=.6)
        ax.set_xticks(np.arange(L), [str(i) for i in range(L)], fontsize=6.3)
        ax.set_xlabel("fine-tuning epoch")
        grid_y(ax)
        panel(ax, letter, title, x=-.25)
    axes[0].set_ylabel("Δ val Acc@1 (points)")
    axes[1].legend(handles=[Line2D([], [], color=BASE, lw=.8, label="one city"), Line2D([], [], color=OURS, lw=1.6, label="median")],
                   loc="lower right", fontsize=6.0)
    save(fig, "fig8_finetune")


# =============================================================================== Figure 9 / 10 (need scripts/run_paper_ablate.sh)
ABL_LABEL = {"noFreq": "visit frequency", "noRecency": "recency", "noTime": "hour + weekday", "noTrans": "last → next",
             "noTrip": "in this trip / is last", "noRetDist": "distance (return)", "noPointer": "latent pointer",
             "noActPath": "activity path", "noExpDist": "distance + anchors (explore)",
             "noEvidence": "all personal evidence (encoder only)",
             "gateNoS": r"exploration-law term $g_S\log S_u$", "gateNoPersonal": "all personal gate terms",
             "gateConst": "one global return/explore prior"}
ABL_GROUP = [("Return evidence removed", ["noFreq", "noRecency", "noTime", "noTrans", "noTrip", "noRetDist", "noPointer"], RET),
             ("Explore evidence removed", ["noActPath", "noExpDist"], EXP),
             ("Gate simplified", ["gateNoS", "gateNoPersonal", "gateConst"], OURS),
             ("No evidence at all", ["noEvidence"], INK2)]


def _seed_reports(prefix):
    return R.reports(prefix)


def _abl_deltas(prefix):
    """{name: (macro delta mean over seeds, sd, [per-city mean deltas])} in points"""
    out = {}
    for rep in _seed_reports(prefix).values():
        base = next((k for k in rep if k.startswith("full_s")), None)
        if base is None:
            continue
        for lab, cells in rep.items():
            if lab.startswith(("_", "full_s")):
                continue
            nm = lab.rsplit("_s", 1)[0]
            cs = [c for c in TGT if c in cells and c in rep[base]]
            if not cs:
                continue
            d = {c: 100 * (cells[c]["acc1"] - rep[base][c]["acc1"]) for c in cs}
            out.setdefault(nm, []).append(d)
    res = {}
    for nm, lst in out.items():
        mac = [np.mean(list(d.values())) for d in lst]
        per = {c: np.mean([d[c] for d in lst if c in d]) for c in TGT if any(c in d for d in lst)}
        res[nm] = (float(np.mean(mac)), float(np.std(mac, ddof=1)) if len(mac) > 1 else 0.0, per)
    return res


def fig9_evidence_ablation():
    D = {**_abl_deltas("paper_R_abl_evidence"), **_abl_deltas("paper_R_abl_gate")}
    if not D:
        print("[skip] fig9: run scripts/run_paper_ablate.sh (stages E G R) and re-run collect.py")
        return
    rows = []
    for title, names, col in ABL_GROUP:
        names = [n for n in names if n in D]
        if names:
            rows.append(("__" + title, None, None, None, col))
            for n in sorted(names, key=lambda n: D[n][0]):
                rows.append((ABL_LABEL.get(n, n),) + D[n] + (col,))
    fig, ax = plt.subplots(figsize=(COL, .19 * len(rows) + .55))
    y = np.arange(len(rows))[::-1]
    lab = []
    for yi, (l, m, sd, per, col) in zip(y, rows):
        if l.startswith("__"):
            lab.append(l[2:])
            continue
        lab.append(l)
        ax.scatter(list(per.values()), [yi] * len(per), s=5, color=MUTED, zorder=2)
        ax.barh(yi, m, color=col, height=.62, zorder=1)
        if sd:
            ax.errorbar(m, yi, xerr=sd, color=INK, lw=.7, capsize=1.5, zorder=3)
    ax.set_yticks(y, lab, fontsize=6.4)
    for t, (l, *_r) in zip(ax.get_yticklabels(), rows):
        if l.startswith("__"):
            t.set_fontweight("bold"); t.set_color(INK)
    ax.axvline(0, color=SOTA, lw=.8)
    ax.set_xlabel(f"Δ zero-shot Acc@1 vs. full GEMMA (points; {N_TGT} cities, 3 seeds)")
    grid_x(ax)
    save(fig, "fig9_evidence_ablation")


def fig10_history():
    reps = _seed_reports("paper_R_hist")
    if not reps:
        print("[skip] fig10: run scripts/run_paper_ablate.sh (stages H R) and re-run collect.py")
        return
    curve = {}
    for rep in reps.values():
        for lab, cells in rep.items():
            if lab.startswith("hist"):
                k = int(lab[4:].rsplit("_s", 1)[0])
                cs = [c for c in TGT if c in cells]
                curve.setdefault(k, []).append(np.mean([cells[c]["acc1"] for c in cs]))
    full = R.agg("zs", TGT)["macro"]
    ks = sorted(curve)
    m = np.array([np.mean(curve[k]) for k in ks]); sd = np.array([np.std(curve[k]) for k in ks])
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    ax.fill_between(ks, m - sd, m + sd, color=OURS, alpha=.2, lw=0)
    ax.plot(ks, m, color=OURS, marker="o", ms=3, lw=1.4, label="GEMMA zero-shot")
    ax.axhline(full[0], color=OURS, ls=":", lw=1, label="GEMMA, full history")
    ax.axhline(np.mean([R.sup("STHGCN", c) for c in TGT]), color=SOTA, lw=1, ls="--",
               label="STHGCN (full history, trained in city)")
    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("past check-ins visible per user (k most recent)")
    ax.set_ylabel(f"macro Acc@1 ({N_TGT} cities)")
    grid_y(ax)
    ax.legend(loc="lower right", fontsize=6.0)
    save(fig, "fig10_history")


ALL = {f.__name__: f for f in (fig1_teaser, fig3_main_forest, fig4_llm, fig5_anatomy, fig6_weights, fig7_ablation,
                               fig8_finetune, fig9_evidence_ablation, fig10_history)}

if __name__ == "__main__":
    setup()
    R = Results()
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
