"""paper/collect.py -- gather every number the paper uses into ONE json (paper/data/results.json).

Reads the experiment logs under their server names (logs/...) and the analysis tables
(analysis_out/tables/*.csv, falling back to the printed tables inside logs/analysis_A*.log).

  python paper/collect.py --logs logs --analysis analysis_out/tables --out paper/data/results.json
  python paper/collect.py --logs paper/logs_snapshot            # offline snapshot shipped with the repo

Runs used by the paper (all GEMMA = v6 HREM recipe, seeds 0/1/2):
  v6_F_s{S}.log        source-city training of the realizer (learned theta / gate per seed)
  v6_E_s{S}.log        target cities: zero_shot  (+ full with train+val visible, kept for the appendix)
  v6_E_llm_s{S}.log    target cities: LLM-Move protocol (gt+rand100, last-15 history)
  paper_V_s{S}.log     target cities: full-shot, train-only visibility (main fine-tuned result)
  paper_S_s{S}.log     source cities: full-shot (in-domain comparison)
  final_R_zero.txt     zero-shot hand-set evidence weights ("hand") vs learned  (theta ablation)
  final_R_ablation.txt representation/backbone ablations (world-model objectives)
  v6_R_diag.txt        test-time evidence-channel drops (pointer / prefix)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

SRC = ["istanbul", "petaling_jaya", "jakarta", "new_york", "kuwait_city", "sao_paulo"]
TGT = ["tokyo", "beijing", "shanghai", "palembang", "melbourne", "sydney", "tangerang", "moscow", "bandung"]

FV = re.compile(
    r"\[(?P<city>[a-z_]+)/(?P<shot>[a-z_]+)(?: \((?P<src>source)\))?\] FULL-VOCAB "
    r"acc@1=(?P<acc1>[\d.]+) acc@5=(?P<acc5>[\d.]+) acc@10=(?P<acc10>[\d.]+) ndcg@5=(?P<ndcg5>[\d.]+) "
    r"mrr=(?P<mrr>[\d.]+) \| next-activity acc=(?P<act_acc>[\d.]+) f1=(?P<act_f1>[\d.]+) \| "
    r"revisit\((?P<rho>[\d.]+)\) acc@1=(?P<ret_acc1>[\d.]+) explore acc@1=(?P<exp_acc1>[\d.]+) \| "
    r"cond: explore@1/@10=(?P<cexp1>[\d.]+)/(?P<cexp10>[\d.]+) revisit@1=(?P<crev1>[\d.]+) "
    r"lat-explore@10=(?P<latexp10>[\d.]+) ptr@1=(?P<ptr1>[\d.]+) pi\(exp/rev\)=(?P<pi_exp>[\d.]+)/(?P<pi_rev>[\d.]+)")
LLM = re.compile(
    r"\[(?P<city>[a-z_]+)/zero_shot\]\s+\|\| LLM-200\[gt\+near100\] acc@1=(?P<near_acc1>[\d.]+)\+-(?P<near_sd>[\d.]+) "
    r"acc@5=(?P<near_acc5>[\d.]+) ndcg@5=(?P<near_ndcg5>[\d.]+) \[near100 (?P<near_only>[\d.]+)\] "
    r"\[gt\+rand100 (?P<rand_acc1>[\d.]+)\]")
W = re.compile(r"\[(?P<who>[a-z_ ]+?)(?:/(?P<shot>[a-z_]+))?\] ft epoch (?P<ep>\d+) loss=(?P<loss>[\d.]+).*?"
               r"ret\[(?P<ret>[^\]]*)\] exp\[(?P<exp>[^\]]*)\] gate\[(?P<gate>[^\]]*)\]")
VAL = re.compile(r"\[(?P<who>[a-z_]+)(?:/(?P<shot>[a-z_]+))?\] val@(?P<stage>[\w-]+): acc@1=(?P<acc1>[\d.]+) "
                 r"acc@5=(?P<acc5>[\d.]+)(?: act@1=(?P<act1>[\d.]+))?")


def kv(s):
    return {k: float(v) for k, v in (t.split("=") for t in s.split())}


def parse_fullvocab(path, run, seed):
    rows = []
    for line in open(path, errors="ignore"):
        m = FV.search(line)
        if m:
            d = m.groupdict()
            r = {k: (float(v) if k not in ("city", "shot", "src") else v) for k, v in d.items() if v is not None}
            r.update(run=run, seed=seed, role="source" if (d["src"] or d["city"] in SRC) else "target")
            r.pop("src", None)
            rows.append(r)
    return rows


def parse_llm(path, seed):
    out = []
    for line in open(path, errors="ignore"):
        m = LLM.search(line)
        if m:
            d = m.groupdict()
            out.append({**{k: float(v) for k, v in d.items() if k != "city"}, "city": d["city"], "seed": seed})
    return out


def parse_weights(path, run, seed):
    out = []
    for line in open(path, errors="ignore"):
        m = W.search(line)
        if m:
            d = m.groupdict()
            out.append({"run": run, "seed": seed, "who": d["who"].strip(), "shot": d["shot"], "epoch": int(d["ep"]),
                        "loss": float(d["loss"]), "ret": kv(d["ret"]), "exp": kv(d["exp"]), "gate": kv(d["gate"])})
    return out


def parse_val_curve(path, run, seed):
    """validation Acc@1 trace during fine-tuning (for the fine-tuning dynamics figure)."""
    out = []
    for line in open(path, errors="ignore"):
        m = VAL.search(line)
        if m:
            out.append({"run": run, "seed": seed, "who": m["who"], "shot": m["shot"], "stage": m["stage"],
                        "acc1": float(m["acc1"]), "acc5": float(m["acc5"])})
    return out


ROW = re.compile(r"^\s*([a-z_]+)\s+(\d+)\s+([\d.]+)\s+\|(.*)$")
CELL = re.compile(r"([\d.]+)/([\d.]+)([<>=])\s*(?:d([+-][\d.]+)(\*)?)?")


def parse_report(path):
    """scripts/final_report.py text tables -> {variant: {city: {acc1, ndcg5, sig, d, dsig}}, 'sthgcn': {...}, 'n': {...}}"""
    lines = open(path, errors="ignore").read().splitlines()
    head = next((l for l in lines if l.strip().startswith("city")), None)
    if head is None:
        return {}
    names = [h.strip() for h in head.split("|")[1:]]
    rep = {n: {} for n in names}
    rep["_sthgcn"], rep["_n"] = {}, {}
    for l in lines:
        m = ROW.match(l)
        if not m:
            continue
        city = m.group(1)
        rep["_n"][city] = int(m.group(2))
        rep["_sthgcn"][city] = float(m.group(3))
        for n, cell in zip(names, m.group(4).split("|")):
            c = CELL.search(cell)
            if c:
                rep[n][city] = {"acc1": float(c.group(1)), "ndcg5": float(c.group(2)), "sig": c.group(3),
                                "d": float(c.group(4)) if c.group(4) else None, "dsig": bool(c.group(5))}
    return rep


def _right_aligned(block):
    """pandas DataFrame.to_string() output is right-aligned: a column ends where EVERY row has a
    token ending at the same position.  Multi-word headers / values ('Petaling Jaya') stay intact."""
    if len(block) < 2:
        return None
    width = max(len(l) for l in block)
    rows = [l.ljust(width) for l in block]
    ends = None
    for l in rows[1:]:
        e = {m.end() for m in re.finditer(r"\S+", l)}
        ends = e if ends is None else ends & e
    ends = sorted(ends or [])
    if not ends:
        return None
    starts = [0] + ends[:-1]
    cut = lambda l: [l[s:e].strip() for s, e in zip(starts, ends)]
    head = cut(rows[0])
    data = []
    for l in rows[1:]:
        vals = []
        for v in cut(l):
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None if v in ("", "NaN", "nan") else v)
        data.append(vals)
    head = [h if h else f"col{i}" for i, h in enumerate(head)]
    return pd.DataFrame(data, columns=head)


def parse_log_tables(path):
    """'[table] NAME' blocks printed by analysis/*.py -> {NAME: DataFrame}"""
    txt = open(path, errors="ignore").read().splitlines()
    out, i = {}, 0
    while i < len(txt):
        if txt[i].startswith("[table] "):
            name = txt[i][8:].strip()
            j = i + 1
            block = []
            while (j < len(txt) and txt[j].strip() and not txt[j].startswith(("[", "/"))
                   and "Warning" not in txt[j]):
                block.append(txt[j])
                j += 1
            df = _right_aligned(block)
            if df is not None:
                out[name] = df
            i = j
        else:
            i += 1
    return out


A4CITY = re.compile(r"^\[(?P<city>[a-z_]+)\] n=(?P<n>\d+) (?P<rest>.*user frequency=.*)$")


def a4_per_city(logs):
    """analysis_A4.log prints single-signal Acc@1 per city -> lets the paper drop any city from the macro"""
    rows = []
    for f in glob.glob(os.path.join(logs, "analysis_A4.log")):
        for line in open(f, errors="ignore"):
            m = A4CITY.match(line.strip())
            if m:
                r = {"city": m["city"], "n": int(m["n"])}
                r.update({k.strip(): float(v) for k, v in re.findall(r"([a-z][a-z \-]*?)=([\d.]+)", m["rest"])})
                rows.append(r)
    return rows


def analysis_tables(adir, logs):
    tabs = {}
    for f in sorted(glob.glob(os.path.join(adir or "", "*.csv"))):
        tabs[os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f)
    for f in sorted(glob.glob(os.path.join(logs, "analysis_A*.log"))):
        for k, v in parse_log_tables(f).items():
            tabs.setdefault(k, v)
    out = {k: json.loads(v.to_json(orient="records")) for k, v in tabs.items()}
    out["A4_per_city"] = a4_per_city(logs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--analysis", default="analysis_out/tables")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "results.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    L = lambda n: os.path.join(a.logs, n)
    R = {"fullvocab": [], "llm": [], "weights": [], "val": [], "reports": {}, "analysis": {}}
    for s in a.seeds:
        for run, f in (("E", f"v6_E_s{s}.log"), ("V", f"paper_V_s{s}.log"), ("S", f"paper_S_s{s}.log")):
            if os.path.exists(L(f)):
                R["fullvocab"] += parse_fullvocab(L(f), run, s)
                R["weights"] += parse_weights(L(f), run, s)
                R["val"] += parse_val_curve(L(f), run, s)
            else:
                print("missing", L(f))
        if os.path.exists(L(f"v6_F_s{s}.log")):
            R["weights"] += parse_weights(L(f"v6_F_s{s}.log"), "F", s)
            R["val"] += parse_val_curve(L(f"v6_F_s{s}.log"), "F", s)
        if os.path.exists(L(f"v6_E_llm_s{s}.log")):
            R["llm"] += parse_llm(L(f"v6_E_llm_s{s}.log"), s)
    for f in glob.glob(L("*_R_*.txt")):
        R["reports"][os.path.splitext(os.path.basename(f))[0]] = parse_report(f)
    R["analysis"] = analysis_tables(a.analysis, a.logs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
    c = lambda run, shot: len([r for r in R["fullvocab"] if r["run"] == run and r["shot"] == shot])
    print(f"wrote {a.out}: zero_shot rows={c('E', 'zero_shot')}  full(train+val)={c('E', 'full')}  "
          f"full(train-only)={c('V', 'full')}  source full={c('S', 'full')}  llm={len(R['llm'])}  "
          f"weights={len(R['weights'])}  reports={sorted(R['reports'])}  analysis={sorted(R['analysis'])}")


if __name__ == "__main__":
    main()
