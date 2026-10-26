"""paper/tables.py -- LaTeX tables (booktabs, ACM sigconf) from paper/data/results.json.

  python paper/tables.py            -> paper/out/tables/*.tex      (\\input{} them in the paper)

tab_datasets      Massive-STEPS cities: role, users, check-ins, POIs, test queries, revisit share, novelty exponent
tab_main_unseen   8 unseen cities (Beijing excluded): 11 supervised baselines + GEMMA zero-shot + GEMMA fine-tuned  (table*)
tab_main_source   6 source cities: same baselines + GEMMA fine-tuned                             (table*)
tab_llm           LLM-agent protocol: 15 published LLM agents vs GEMMA zero-shot                  (table*)
tab_fairness      fine-tuned visibility check (train+val vs train-only) and aggregate scoreboard   (column)
tab_ablation      learned vs hand-set weights, backbone objectives, (+ evidence/gate drops if run)  (column)
tab_weights       learned evidence weights and gate coefficients                                   (column)
Packages needed in the paper: booktabs, multirow, makecell (optional), xcolor (for \\rowcolor), amssymb.
"""
import os

import numpy as np

from baselines import LLM_ZS, SUPERVISED
from data_api import Results
from style import HERE, SRC, TGT, pretty

OUT = os.path.join(HERE, "out", "tables")
SUP_ORDER = ["FPMC", "RNN", "LSTPM", "DeepMove", "Flashback", "Graph-Flashback", "LoTNext", "GETNext", "STHGCN",
             "GNPR-SID", "UniMove"]
WORDS = {6: "six", 7: "seven", 8: "eight", 9: "nine"}
ABBR = {"petaling_jaya": "P. Jaya", "kuwait_city": "Kuwait C.", "sao_paulo": "São Paulo", "new_york": "New York",
        "tangerang": "Tangerang", "palembang": "Palembang", "melbourne": "Melbourne"}


def cname(c):
    return ABBR.get(c, pretty(c))


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.2f}"


def mark(vals):
    """rank per column: returns {row_index: 'best'|'second'}"""
    v = [(round(x, 3), i) for i, x in enumerate(vals) if x is not None and not np.isnan(x)]   # ties at print precision
    if not v:
        return {}
    xs = sorted({x for x, _ in v}, reverse=True)
    out = {i: "best" for x, i in v if x == xs[0]}
    if len(xs) > 1:
        out.update({i: "second" for x, i in v if x == xs[1]})
    return out


def fmt(x, tag=None):
    s = f3(x)
    if tag == "best":
        return f"\\textbf{{{s}}}"
    if tag == "second":
        return f"\\underline{{{s}}}"
    return s


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(body)
    print("[table]", os.path.join(OUT, name + ".tex"))


# ------------------------------------------------------------------------------------------------ datasets
def tab_datasets(R):
    A1 = {r["city"]: r for r in R.table("A1_mobility_laws")}
    rho = {}
    for st in ("zs", "src"):
        for c, v in R.city(st, "rho").items():
            rho[c] = v[0]
    lines = []
    for grp, cities in (("Source (training)", SRC), ("Unseen (zero-shot)", TGT)):
        lines.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\textit{{{grp}}}}} \\\\")
        for c in cities:
            r = A1.get(pretty(c), {})
            lines.append(f"{pretty(c)} & {int(r.get('users', 0)):,} & {int(r.get('check-ins', 0)):,} & {int(r.get('POIs', 0)):,} & "
                         f"{R.n.get(c, 0):,} & {r.get('revisit_share', float('nan')):.2f} & {rho.get(c, float('nan')):.2f} & "
                         f"{r.get('gamma (Pnew~S^-g)', float('nan')):.2f} \\\\")
    body = r"""\begin{table}[t]
\caption{Massive-STEPS cities. \emph{Queries}: test next-visit queries kept by the STHGCN protocol (target POI and user
seen in training). \emph{Revisit}: share of check-ins to an already visited place, over all check-ins and over the
scored queries. $\gamma$: novelty exponent of $P(\text{new})\propto S^{-\gamma}$.}
\label{tab:datasets}
\small\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrrccc}
\toprule
City & Users & Check-ins & POIs & Queries & \multicolumn{2}{c}{Revisit} & $\gamma$ \\
 & & & & & all & test & \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_datasets", body)


# ------------------------------------------------------------------------------------------------ supervised comparison
def _main(R, cities, rows_gemma, name, caption, label):
    """rows_gemma: list of (label, setting, uses_target_data)"""
    rows = []
    for m in SUP_ORDER:
        acc = [SUPERVISED[m][c][0] for c in cities]
        nd = [SUPERVISED[m][c][1] for c in cities]
        rows.append((m, acc, nd, True, None))
    for lab, st, tgt in rows_gemma:
        ca, cn = R.city(st, "acc1"), R.city(st, "ndcg5")
        rows.append((lab, [ca[c][0] for c in cities], [cn[c][0] for c in cities], tgt, [ca[c][1] for c in cities]))
    # averages
    w = np.array([R.n[c] for c in cities], float)
    table = []
    for lab, acc, nd, tgt, sd in rows:
        table.append((lab, acc + [float(np.mean(acc)), float(np.dot(w, acc) / w.sum())], nd + [float(np.mean(nd))], tgt))
    ncol_a = len(cities) + 2
    marks_a = [mark([r[1][j] for r in table]) for j in range(ncol_a)]
    marks_n = [mark([r[2][j] for r in table]) for j in range(len(cities) + 1)]
    head1 = " & ".join(f"\\multicolumn{{2}}{{c}}{{{cname(c)}}}" for c in cities)
    head2 = " & ".join(["A@1 & N@5"] * len(cities))
    lines = []
    for i, (lab, acc, nd, tgt) in enumerate(table):
        cells = []
        for j, c in enumerate(cities):
            cells += [fmt(acc[j], marks_a[j].get(i)), fmt(nd[j], marks_n[j].get(i))]
        cells += [fmt(acc[-2], marks_a[-2].get(i)), fmt(nd[-1], marks_n[-1].get(i)), fmt(acc[-1], marks_a[-1].get(i))]
        tick = r"$\checkmark$" if tgt else r"$\times$"
        pre = "\\rowcolor{gemmarow}" if lab.startswith("GEMMA") else ""
        if lab.startswith("GEMMA") and not table[i - 1][0].startswith("GEMMA"):
            lines.append("\\midrule")
        lines.append(f"{pre}{lab} & {tick} & " + " & ".join(cells) + " \\\\")
    cols = "l c " + " ".join(["cc"] * len(cities)) + " ccc"
    body = f"""\\begin{{table*}}[t]
\\caption{{{caption}}}
\\label{{{label}}}
\\scriptsize\\setlength{{\\tabcolsep}}{{1.7pt}}
\\begin{{tabular}}{{{cols}}}
\\toprule
 & Target & {head1} & \\multicolumn{{3}}{{c}}{{Average}} \\\\
\\cmidrule(lr){{3-{2 + 2 * len(cities)}}}\\cmidrule(lr){{{3 + 2 * len(cities)}-{5 + 2 * len(cities)}}}
Model & train & {head2} & A@1 & N@5 & A@1$_{{\\text{{pooled}}}}$ \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    write(name, body)


def tab_main_unseen(R):
    _main(R, TGT, [("GEMMA zero-shot", "zs", False), ("GEMMA fine-tuned", "ft", True)], "tab_main_unseen",
          f"Next-POI prediction on the {WORDS[len(TGT)]} cities GEMMA never saw during training (STHGCN protocol; baselines from "
          "Massive-STEPS). \\emph{Target train}: the model is trained on that city's training split. GEMMA zero-shot "
          "uses no data from the city except each test user's own past check-ins; fine-tuned GEMMA uses the training "
          "split only. Mean of 3 seeds. Best in \\textbf{bold}, second \\underline{underlined}. A@1 = Acc@1, "
          "N@5 = NDCG@5; pooled = weighted by the number of test queries.", "tab:main-unseen")


def tab_main_source(R):
    _main(R, SRC, [("GEMMA fine-tuned", "src", True)], "tab_main_source",
          "Next-POI prediction on the six source cities GEMMA was trained on (in-domain; fine-tuned GEMMA, 3 seeds).",
          "tab:main-source")


# ------------------------------------------------------------------------------------------------ LLM protocol
def tab_llm(R):
    gp = R.llm_proto("rand_acc1")
    rows = [(k, [LLM_ZS[k][c][0] for c in TGT], None) for k in LLM_ZS]
    rows.append(("GEMMA zero-shot (ours)", [gp[c][0] for c in TGT], [gp[c][1] for c in TGT]))
    marks = [mark([r[1][j] for r in rows]) for j in range(len(TGT))]
    avg = [float(np.mean(r[1])) for r in rows]
    mavg = mark(avg)
    lines, cur = [], None
    for i, (k, v, sd) in enumerate(rows):
        fw = k.split(" (")[0] if "(" in k else ""
        model = k.split("(")[1].rstrip(")") if "(" in k else k
        show = fw
        if fw != cur:
            lines.append("\\midrule")
            cur = fw
        else:
            show = ""
        pre = "\\rowcolor{gemmarow}" if k.startswith("GEMMA") else ""
        if k.startswith("GEMMA"):
            show, model = "\\textbf{GEMMA}", "zero-shot, no LLM"
        cells = [fmt(x, marks[j].get(i)) for j, x in enumerate(v)]
        lines.append(f"{pre}{show} & {model} & " + " & ".join(cells) + f" & {fmt(avg[i], mavg.get(i))} \\\\")
    body = r"""\begin{table*}[t]
\caption{Zero-shot comparison with LLM agents under the LLM-Move protocol of Massive-STEPS: candidates are the true next
place plus 100 random places of the city, history = the user's last 15 trajectories, Acc@1. LLM rows are the published
numbers (200 users); GEMMA is the mean over 3 seeds $\times$ 5 user samples of 200 users. GEMMA uses no LLM at inference.}
\label{tab:llm}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll""" + "c" * len(TGT) + r"""c}
\toprule
Framework & Model & """ + " & ".join(cname(c) for c in TGT) + r""" & Avg. \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_llm", body)


# ------------------------------------------------------------------------------------------------ fairness / scoreboard
def tab_fairness(R):
    def row(lab, st):
        a, n = R.agg(st, TGT), R.agg(st, TGT, "ndcg5")
        return (f"{lab} & {a['macro'][0]:.4f} & {a['pooled'][0]:.4f} & {n['macro'][0]:.4f} \\\\")
    sa, sn = R.sup_agg("STHGCN", TGT), R.sup_agg("STHGCN", TGT, "ndcg5")
    ga, gn = R.sup_agg("GETNext", TGT), R.sup_agg("GETNext", TGT, "ndcg5")
    body = r"""\begin{table}[t]
\caption{Aggregate results on the """ + WORDS[len(TGT)] + r""" unseen cities. Fine-tuned GEMMA is shown with the test-time
history drawn from the training split only (as STHGCN) and from train+validation.}
\label{tab:aggregate}
\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{2}{c}{Acc@1} & NDCG@5 \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}
Model & macro & pooled & macro \\
\midrule
""" + f"GETNext & {ga['macro']:.4f} & {ga['pooled']:.4f} & {gn['macro']:.4f} \\\\\n" \
        + f"STHGCN & {sa['macro']:.4f} & {sa['pooled']:.4f} & {sn['macro']:.4f} \\\\\n\\midrule\n" \
        + row("GEMMA zero-shot", "zs") + "\n" + row("GEMMA FT (train)", "ft") + "\n" + row("GEMMA FT (train+val)", "ft_tv") + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_fairness", body)


# ------------------------------------------------------------------------------------------------ ablation
def tab_ablation(R):
    noB = TGT
    lines = []
    rz = R.report("final_R_zero")
    if rz.get("hand"):
        v = np.mean([rz["v6"][c]["acc1"] for c in noB]); h = np.mean([rz["hand"][c]["acc1"] for c in noB])
        lines.append(f"Learned evidence weights $\\to$ hand-set & {100 * (h - v):+.1f} \\\\")
    ra = R.report("final_R_ablation")
    lab = {"noWM": "no world-model objective", "noJEPA": "no JEPA objective", "randomInit": "no backbone pretraining",
           "flat": "flat activity taxonomy", "noPtr": "no latent pointer"}
    if ra.get("full"):
        f = np.mean([ra["full"][c]["acc1"] for c in noB])
        for k, l in lab.items():
            if k in ra:
                lines.append(f"{l} & {100 * (np.mean([ra[k][c]['acc1'] for c in noB]) - f):+.1f} \\\\")
    # optional test-time ablations from scripts/run_paper_ablate.sh
    for prefix, title in (("paper_R_abl_evidence", "Remove evidence (test time)"), ("paper_R_abl_gate", "Simplify gate (test time)")):
        reps = [v for k, v in R.R["reports"].items() if k.startswith(prefix)]
        if not reps:
            continue
        lines.append(f"\\midrule\n\\multicolumn{{2}}{{l}}{{\\textit{{{title}}}}} \\\\")
        names = sorted({l.rsplit('_s', 1)[0] for rep in reps for l in rep if not l.startswith(("_", "full_s"))})
        for nm in names:
            ds = []
            for rep in reps:
                base = next(k for k in rep if k.startswith("full_s"))
                l = next((k for k in rep if k.rsplit("_s", 1)[0] == nm), None)
                if l:
                    ds.append(100 * np.mean([rep[l][c]["acc1"] - rep[base][c]["acc1"] for c in noB if c in rep[l]]))
            lines.append(f"{nm} & {np.mean(ds):+.1f} \\\\")
    body = r"""\begin{table}[t]
\caption{Ablations: change of zero-shot macro Acc@1 (points, """ + WORDS[len(TGT)] + r""" unseen cities).}
\label{tab:ablation}
\small
\begin{tabular}{lr}
\toprule
Variant & $\Delta$Acc@1 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_ablation", body)


# ------------------------------------------------------------------------------------------------ weights
def tab_weights(R):
    from figures import EXP_NAMES, GATE_NAMES, RET_NAMES
    src = R.source_weights()
    ftv = R.finetuned_weights("V")
    lines = []
    for part, names, title in (("ret", RET_NAMES, "Return evidence $\\mathbf{w}_0$"), ("exp", EXP_NAMES, "Explore evidence $\\mathbf{w}'_0$"),
                               ("gate", GATE_NAMES, "Gate")):
        lines.append(f"\\midrule\n\\multicolumn{{3}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for k, l in names.items():
            s = np.array([w[part][k] for w in src])
            f = np.array([w[part][k] for w in ftv.values()])
            l = l.replace("†", "$^\\dagger$").replace("#", "\\#").replace("→", "$\\to$")
            lines.append(f"{l} & {s.mean():.2f} $\\pm$ {s.std(ddof=1):.2f} & {f.min():.2f} -- {f.max():.2f} \\\\")
    body = r"""\begin{table}[t]
\caption{Learned parameters (at $\mathbf{z}=0$). Source: trained on six cities, mean $\pm$ sd over 3 seeds.
Fine-tuned: range over the """ + WORDS[len(TGT)] + r""" unseen cities $\times$ 3 seeds after fine-tuning. $^\dagger$\,needs the target-city log,
inactive in zero-shot.}
\label{tab:weights}
\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\toprule
Parameter & Source-trained & Fine-tuned (range) \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_weights", body)


if __name__ == "__main__":
    import style
    style.setup()
    R = Results()
    for f in (tab_datasets, tab_main_unseen, tab_main_source, tab_llm, tab_fairness, tab_ablation, tab_weights):
        f(R)
