"""Hyper-parameter search for GEMMA world-model pretraining (single GPU, sequential trials).

Selection is LEAKAGE-FREE: every trial pretrains on the source cities MINUS the dev
city/cities and is scored zero-shot on the dev cities' VALIDATION split under the STHGCN
protocol.  The 9 held-out test cities are never touched during tuning.

    python scripts/tune_pretrain.py --base_run runs/gemma_v4 --data_root data \
        --sources istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
        --dev kuwait_city --trials 12 --epochs 2 --qpc 25000
    # -> runs/tune/trials.jsonl (every trial) and runs/tune/best.json (flags of the best)

Objective = mean over dev cities of  Acc@1 + 0.25 * NDCG@5.
Uses Optuna (TPE) when installed (pip install optuna), otherwise seeded random search.
"""
import argparse, json, os, random, shutil, subprocess, sys, time

PY = sys.executable
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPACE = {   # name: (kind, choices or (low, high))
    "init":             ("cat", ["jepa_v4", "random"]),
    "lr_backbone":      ("log", (5e-5, 5e-4)),
    "lr_head":          ("log", (3e-4, 3e-3)),
    "lambda_wm_jepa":   ("cat", [0.25, 0.5, 1.0, 2.0]),
    "lambda_wm_ptr":    ("cat", [0.3, 1.0, 2.0]),
    "lambda_wm_lat":    ("cat", [0.25, 0.5, 1.0]),
    "lambda_wm_motif":  ("cat", [0.0, 0.1, 0.3, 1.0]),
    "lambda_wm_dt":     ("cat", [0.0, 0.1, 0.3]),
    "wm_H":             ("cat", [1, 2, 3]),
    "wm_infill":        ("cat", [0.0, 0.15, 0.3]),
    "ema_decay":        ("cat", [0.99, 0.996, 0.999]),
}


def sample_random(rng):
    hp = {}
    for k, (kind, v) in SPACE.items():
        if kind == "cat":
            hp[k] = rng.choice(v)
        else:
            lo, hi = v
            hp[k] = float(10 ** rng.uniform(__import__("math").log10(lo), __import__("math").log10(hi)))
    return hp


def sample_optuna(trial):
    hp = {}
    for k, (kind, v) in SPACE.items():
        hp[k] = trial.suggest_categorical(k, v) if kind == "cat" else trial.suggest_float(k, *v, log=True)
    return hp


def hp_flags(hp):
    """Flags for train_realizer_source.py (infill weight tied to 0.5 x JEPA weight)."""
    f = ["--lr_backbone", f"{hp['lr_backbone']:.3g}", "--lr_head", f"{hp['lr_head']:.3g}",
         "--lambda_wm_jepa", str(hp["lambda_wm_jepa"]), "--lambda_wm_infill", str(0.5 * hp["lambda_wm_jepa"]),
         "--lambda_wm_ptr", str(hp["lambda_wm_ptr"]), "--lambda_wm_lat", str(hp["lambda_wm_lat"]),
         "--lambda_wm_motif", str(hp["lambda_wm_motif"]), "--lambda_wm_dt", str(hp["lambda_wm_dt"]),
         "--wm_H", str(hp["wm_H"]), "--wm_infill", str(hp["wm_infill"]), "--ema_decay", str(hp["ema_decay"])]
    if hp["init"] == "random":
        f.append("--random_backbone")
    return f


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n"); fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)


def evaluate_trial(a, k, hp):
    d = os.path.join(a.out, f"t{k:02d}")
    os.makedirs(d, exist_ok=True)
    for f in ("config.json", "gemma.pt"):
        shutil.copy(os.path.join(a.base_run, f), d)
    log = os.path.join(d, "trial.log")
    srcs = [c for c in a.sources if c not in a.dev]
    budget = ["--epochs", str(a.epochs), "--min_epochs", str(a.epochs), "--patience", "99",
              "--max_queries_per_city", str(a.qpc), "--batch", "128", "--val_per_city", "1000",
              "--seed", str(a.seed)]
    rc = run([PY, "scripts/train_realizer_source.py", "--run_dir", d, "--data_root", a.data_root,
              "--cities", *srcs, "--retriever", "wm", "--tune_backbone", *budget, *hp_flags(hp),
              *a.extra.split()], log)
    if rc != 0:
        return None, {"error": f"train rc={rc}"}
    ev = [PY, "scripts/run_downstream.py", "--run_dir", d, "--data_root", a.data_root,
          "--cities", *a.dev, "--retriever", "wm", "--shots", "zero_shot", "--protocols", "full",
          "--eval_split", "validation", "--eval_protocol", "sthgcn", "--out_tag", "_tune",
          *a.extra.split()]
    if hp["init"] == "random":
        ev.append("--random_backbone")
    rc = run(ev, log)
    if rc != 0:
        return None, {"error": f"eval rc={rc}"}
    res = {}
    for c in a.dev:
        m = json.load(open(os.path.join(d, f"wm_{c}_zero_shot_tune.json")))
        res[c] = {"acc@1": m["acc@1"], "ndcg@5": m["ndcg@5"], "act@1": m.get("act@1"),
                  "n": m.get("n_test_scored")}
    obj = sum(r["acc@1"] + 0.25 * r["ndcg@5"] for r in res.values()) / len(res)
    return obj, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", default="runs/gemma_v4"); ap.add_argument("--data_root", default="data")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", default=["kuwait_city"])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--qpc", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tune")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--extra", default="", help="extra flags for both scripts, e.g. '--allow_cpu'")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jl = os.path.join(a.out, "trials.jsonl")
    done = [json.loads(l) for l in open(jl)] if os.path.exists(jl) else []
    print(f"{len(done)} finished trials found in {jl}")

    def record(k, hp, obj, res, dt):
        row = {"trial": k, "hp": hp, "objective": obj, "dev": res, "minutes": round(dt / 60, 1)}
        with open(jl, "a") as f:
            f.write(json.dumps(row) + "\n")
        done.append(row)
        ok = [r for r in done if r["objective"] is not None]
        if ok:
            best = max(ok, key=lambda r: r["objective"])
            json.dump({"hp": best["hp"], "flags": hp_flags(best["hp"]), "objective": best["objective"],
                       "trial": best["trial"], "dev": best["dev"]},
                      open(os.path.join(a.out, "best.json"), "w"), indent=2)
        print(f"trial {k}: objective={obj} dev={res} ({dt / 60:.0f} min)", flush=True)

    try:
        import optuna
        sampler = optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=min(5, a.trials))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        for r in done:                                  # warm start from finished trials
            if r["objective"] is not None:
                study.add_trial(optuna.trial.create_trial(
                    params=r["hp"], value=r["objective"],
                    distributions={k: (optuna.distributions.CategoricalDistribution(v) if kind == "cat"
                                       else optuna.distributions.FloatDistribution(*v, log=True))
                                   for k, (kind, v) in SPACE.items()}))
        use_optuna = True
    except ImportError:
        print("optuna not installed -> seeded random search (pip install optuna for TPE)")
        use_optuna = False
    rng = random.Random(a.seed + len(done))
    for k in range(len(done), a.trials):
        if use_optuna:
            trial = study.ask(); hp = sample_optuna(trial)
        else:
            hp = sample_random(rng)
        if a.dry_run:
            print(k, hp, " ".join(hp_flags(hp))); continue
        t0 = time.time()
        obj, res = evaluate_trial(a, k, hp)
        if use_optuna:
            study.tell(trial, obj if obj is not None else float("-inf"))
        record(k, hp, obj, res, time.time() - t0)
    ok = [r for r in done if r.get("objective") is not None]
    print("\n=== leaderboard (dev = %s, validation split, STHGCN protocol) ===" % ",".join(a.dev))
    for r in sorted(ok, key=lambda r: -r["objective"])[:10]:
        print(f"t{r['trial']:02d} obj={r['objective']:.4f} " + " ".join(f"{k}={v}" for k, v in r["hp"].items()))


if __name__ == "__main__":
    main()
