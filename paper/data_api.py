"""paper/data_api.py -- typed access to paper/data/results.json + the published baselines.

Every number in every figure/table is computed here, once, so the text, tables and figures agree.
Conventions
  * GEMMA numbers are means over seeds {0,1,2}; `sd` is the across-seed standard deviation.
  * Per-city 95% CI of a GEMMA Acc@1:  1.96 * sqrt(p(1-p)/n + sd^2/k)   (test sampling + seed spread).
    STHGCN is a published point estimate without a CI, so the Delta-CI uses GEMMA's CI only -- the same
    convention as the z-test in scripts/final_report.py.
  * macro = unweighted mean over cities; pooled (= micro) = query-weighted mean, n = test queries kept by
    the STHGCN protocol (from the report tables).
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baselines import LLM_ZS, SUPERVISED  # noqa: E402
from style import EXCLUDED, PRETTY, SRC, TGT  # noqa: E402

RES = os.environ.get("GEMMA_RESULTS", os.path.join(HERE, "data", "results.json"))


class Results:
    def __init__(self, path=RES):
        self.R = json.load(open(path))
        rep = self.R["reports"]
        self.n = {}
        for r in rep.values():
            self.n.update(r.get("_n", {}))

    # ------------------------------------------------------------------ GEMMA full-vocabulary metrics
    def rows(self, setting):
        """setting: 'zs' (target zero-shot), 'ft' (target full, train-only = main),
        'ft_tv' (target full, train+val), 'src' (source full)"""
        run, shot = {"zs": ("E", "zero_shot"), "ft": ("V", "full"), "ft_tv": ("E", "full"),
                     "src": ("S", "full")}[setting]
        return [r for r in self.R["fullvocab"] if r["run"] == run and r["shot"] == shot and r["city"] not in EXCLUDED]

    def city(self, setting, metric="acc1"):
        """{city: (mean, sd, k)} over seeds"""
        d = defaultdict(list)
        for r in self.rows(setting):
            d[r["city"]].append(r[metric])
        return {c: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) for c, v in d.items()}

    def ci(self, setting, city, metric="acc1"):
        m, sd, k = self.city(setting, metric)[city]
        n = self.n.get(city, 1000)
        return 1.96 * math.sqrt(m * (1 - m) / n + (sd ** 2) / max(k, 1))

    def agg(self, setting, cities, metric="acc1"):
        """per-seed macro & pooled, then mean +- sd over seeds"""
        by_seed = defaultdict(dict)
        for r in self.rows(setting):
            if r["city"] in cities:
                by_seed[r["seed"]][r["city"]] = r[metric]
        mac, poo = [], []
        for s, d in by_seed.items():
            if len(d) != len(cities):
                continue
            mac.append(np.mean([d[c] for c in cities]))
            w = np.array([self.n[c] for c in cities], float)
            poo.append(float(np.dot(w, [d[c] for c in cities]) / w.sum()))
        f = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        return {"macro": f(mac), "pooled": f(poo), "k": len(mac)}

    # ------------------------------------------------------------------ baselines
    @staticmethod
    def sup(model, city, metric="acc1"):
        return SUPERVISED[model][city][0 if metric == "acc1" else 1]

    def sup_agg(self, model, cities, metric="acc1"):
        v = [self.sup(model, c, metric) for c in cities]
        w = np.array([self.n[c] for c in cities], float)
        return {"macro": float(np.mean(v)), "pooled": float(np.dot(w, v) / w.sum())}

    @staticmethod
    def llm_best(city, framework=None, metric="acc1"):
        """best published LLM agent per city (oracle over the 15 rows, or within one framework)"""
        rows = {k: v for k, v in LLM_ZS.items() if framework is None or k.startswith(framework + " ")}
        k = max(rows, key=lambda m: rows[m][city][0 if metric == "acc1" else 1])
        return rows[k][city][0 if metric == "acc1" else 1], k

    # ------------------------------------------------------------------ LLM protocol (gt + 100 random)
    def llm_proto(self, key="rand_acc1"):
        d = defaultdict(list)
        for r in self.R["llm"]:
            if r["city"] not in EXCLUDED:
                d[r["city"]].append(r[key])
        return {c: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) for c, v in d.items()}

    # ------------------------------------------------------------------ learned parameters
    def source_weights(self):
        """last printed epoch of source training per seed -> list of dicts"""
        last = {}
        for w in self.R["weights"]:
            if w["run"] == "F":
                last[w["seed"]] = w
        return [last[s] for s in sorted(last)]

    def finetuned_weights(self, run):
        """final epoch per (seed, city) of target ('V') or source ('S') fine-tuning"""
        last = {}
        for w in self.R["weights"]:
            if w["run"] == run and w["shot"] == "full" and w["who"] not in EXCLUDED:
                last[(w["seed"], w["who"])] = w
        return last

    def val_curves(self, run):
        """{(seed, city): [(stage, acc1)]} -- only end-of-epoch points + init"""
        d = defaultdict(list)
        for v in self.R["val"]:
            if v["run"] == run and v["who"] not in EXCLUDED and (v["stage"] == "init" or v["stage"].endswith("-end")):
                d[(v["seed"], v["who"])].append((v["stage"], v["acc1"]))
        return d

    def report(self, name):
        rep = self.R["reports"].get(name, {})
        return {k: {c: v for c, v in d.items() if c not in EXCLUDED} for k, d in rep.items()}

    def reports(self, prefix):
        return {k: self.report(k) for k in self.R["reports"] if k.startswith(prefix)}

    def table(self, name):
        """analysis tables (rows keyed by a pretty city name) without the excluded cities"""
        bad = {PRETTY[c] for c in EXCLUDED} | set(EXCLUDED)
        rows = self.R["analysis"].get(name, [])
        return [r for r in rows if r.get("city", r.get("target city")) not in bad]


def ztest(p, p0, n):
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (p - p0) / se
