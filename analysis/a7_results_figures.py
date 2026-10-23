"""A7 — Result figures/tables from runs that already exist (no model is re-run).

Reads (whatever exists; missing pieces are skipped):
  runs/gemma_v4/hrem_<city>_zero_shot_v6.json          GEMMA zero-shot, STHGCN protocol (stage B)
  runs/abl_incity/hrem_<city>_zero_shot_hand.json      hand-set weights, same protocol
  runs/<run>/hrem_<city>_zero_shot_LLM15.json          LLM-Move protocol (gt + 100 random, 15 traj.)
  runs/<run>/ranks_hrem_<city>_zero_shot_G2h{0,1,3}.npy + _G0  history-budget curve (all-targets protocol)
  runs/scale_{1,3,6}/ranks_hrem_<city>_zero_shot_G3.npy       source-scaling curve
Outputs: tables A7_main_zero_shot, A7_llm_protocol; figures A7a_vs_sthgcn, A7b_vs_llm,
         A7c_history_budget, A7d_source_scaling, A7e_revisit_vs_gap
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import TGT, base_args, save_table, setup_mpl, save_fig, pretty
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.sota_ref import SUP, ZS, ZS_GEMINI


def jl(path):
    return json.load(open(path)) if os.path.exists(path) else None


def acc_of(path):
    return float((np.load(path) < 1).mean()) if os.path.exists(path) else float("nan")


def main():
    ap = base_args(__doc__, TGT)
    ap.add_argument("--gemma_run", default="runs/gemma_v4"); ap.add_argument("--hand_run", default="runs/abl_incity")
    ap.add_argument("--nojepa_run", default="runs/abl_random")
    a = ap.parse_args()
    plt = setup_mpl()
    rows = []
    for c in a.cities:
        g = jl(f"{a.gemma_run}/hrem_{c}_zero_shot_v6.json"); h = jl(f"{a.hand_run}/hrem_{c}_zero_shot_hand.json")
        if g is None:
            continue
        n = g.get("n_test_scored", float("nan")); p = g["acc@1"]; ref = SUP.get(c, (float("nan"),))[0]
        se = math.sqrt(p * (1 - p) / n) if n and n == n else float("nan")
        rows.append({"city": pretty(c), "n": n, "STHGCN (supervised)": ref, "GEMMA zero-shot": p,
                     "95% CI +-": 1.96 * se, "z vs STHGCN": (p - ref) / se if se else float("nan"),
                     "hand-set": h["acc@1"] if h else float("nan"), "GEMMA NDCG@5": g["ndcg@5"],
                     "revisit share": g.get("revisit_frac", float("nan"))})
    if rows:
        T = pd.DataFrame(rows)
        T = pd.concat([T, pd.DataFrame([{"city": "macro", **{k: T[k].mean() for k in T.columns if k not in ("city", "n")}}])],
                      ignore_index=True)
        save_table(T, a.out, "A7_main_zero_shot",
                   "Zero-shot GEMMA (no target-city training) vs. supervised STHGCN on the STHGCN test protocol.")
        D = T.iloc[:-1]
        fig, ax = plt.subplots(figsize=(5.2, 2.3))
        x = np.arange(len(D)); w = 0.27
        ax.bar(x - w, D["STHGCN (supervised)"], w, label="STHGCN (trained in city)", color="#999999")
        ax.bar(x, D["GEMMA zero-shot"], w, yerr=D["95% CI +-"], label="GEMMA zero-shot", color="#0072B2",
               error_kw={"lw": 0.6})
        ax.bar(x + w, D["hand-set"], w, label="GEMMA hand-set (no learning)", color="#56B4E9")
        ax.set_xticks(x); ax.set_xticklabels(D["city"], rotation=30, ha="right"); ax.set_ylabel("Acc@1")
        ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        save_fig(fig, a.out, "A7a_vs_sthgcn"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(2.8, 2.4))
        gap = D["GEMMA zero-shot"] - D["STHGCN (supervised)"]
        ax.scatter(D["revisit share"], gap, color="#0072B2")
        for _, r in D.iterrows():
            ax.annotate(r["city"], (r["revisit share"], r["GEMMA zero-shot"] - r["STHGCN (supervised)"]), fontsize=6)
        ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("share of revisit targets"); ax.set_ylabel("GEMMA − STHGCN (Acc@1)")
        save_fig(fig, a.out, "A7e_revisit_vs_gap"); plt.close(fig)
    rows = []
    for c in a.cities:
        g = jl(f"{a.gemma_run}/hrem_{c}_zero_shot_LLM15.json"); h = jl(f"{a.hand_run}/hrem_{c}_zero_shot_LLM15.json")
        if g is None or "llm_gtrand100_acc@1" not in g:
            continue
        rows.append({"city": pretty(c), "LLM-Move (Gemini-2.0)": ZS_GEMINI.get(c, float("nan")),
                     "LLM-Move (best of 4)": ZS.get(c, (float("nan"),))[0],
                     "GEMMA hand-set": h["llm_gtrand100_acc@1"] if h else float("nan"),
                     "GEMMA": g["llm_gtrand100_acc@1"], "GEMMA std": g.get("llm_gtrand100_acc@1_std", float("nan"))})
    if rows:
        L = pd.DataFrame(rows)
        L = pd.concat([L, pd.DataFrame([{"city": "mean", **{k: L[k].mean() for k in L.columns if k != "city"}}])], ignore_index=True)
        save_table(L, a.out, "A7_llm_protocol", "Massive-STEPS LLM-Move protocol (ground truth + 100 random candidates, "
                   "last 15 trajectories as history), Acc@1.")
        D = L.iloc[:-1]
        fig, ax = plt.subplots(figsize=(5.2, 2.2))
        x = np.arange(len(D)); w = 0.27
        ax.bar(x - w, D["LLM-Move (best of 4)"], w, label="LLM-Move (best LLM)", color="#999999")
        ax.bar(x, D["GEMMA hand-set"], w, label="GEMMA hand-set", color="#56B4E9")
        ax.bar(x + w, D["GEMMA"], w, label="GEMMA", color="#0072B2")
        ax.set_xticks(x); ax.set_xticklabels(D["city"], rotation=30, ha="right"); ax.set_ylabel("Acc@1")
        ax.legend(frameon=False, ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        save_fig(fig, a.out, "A7b_vs_llm"); plt.close(fig)
    # history budget (G2, all-targets protocol)
    ks = [("0", "_G2h0"), ("1", "_G2h1"), ("3", "_G2h3"), ("all", "_G0")]
    curves = {}
    for lab, rd in (("GEMMA", a.gemma_run), ("no JEPA", a.nojepa_run), ("hand-set", a.hand_run)):
        ys = [np.nanmean([acc_of(f"{rd}/ranks_hrem_{c}_zero_shot{t}.npy") for c in a.cities]) for _, t in ks]
        if np.isfinite(ys).any():
            curves[lab] = ys
    if curves:
        fig, ax = plt.subplots(figsize=(2.8, 2.3))
        for (lab, ys), col in zip(curves.items(), ("#0072B2", "#E69F00", "#56B4E9")):
            ax.plot(range(len(ks)), ys, "o-", label=lab, color=col)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([k for k, _ in ks])
        ax.set_xlabel("past trajectories available"); ax.set_ylabel("macro Acc@1 (all targets)"); ax.legend(frameon=False)
        save_fig(fig, a.out, "A7c_history_budget"); plt.close(fig)
    ys = [np.nanmean([acc_of(f"runs/scale_{n}/ranks_hrem_{c}_zero_shot_G3.npy") for c in a.cities]) for n in (1, 3, 6)]
    if np.isfinite(ys).any():
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        ax.plot([1, 3, 6], ys, "o-", color="#0072B2"); ax.set_xscale("log"); ax.set_xticks([1, 3, 6])
        ax.set_xticklabels(["1", "3", "6"]); ax.set_xlabel("source cities"); ax.set_ylabel("macro Acc@1 (zero-shot)")
        save_fig(fig, a.out, "A7d_source_scaling"); plt.close(fig)


if __name__ == "__main__":
    main()
