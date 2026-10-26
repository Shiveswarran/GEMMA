"""Shared helpers for the GEMMA data-analysis suite (tables + figures for the paper).

Every script:  python analysis/<script>.py --run_dir runs/gemma_v4 --data_root data --out analysis_out
Outputs go to  <out>/tables/<name>.csv|.tex  and  <out>/figures/<name>.pdf|.png
The light loader below reads the Massive-STEPS CSVs directly (no POI encoder, no GPU) and
uses the same trajectory rules and split files as the experiments.
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

SRC = ["istanbul", "petaling_jaya", "jakarta", "new_york", "kuwait_city", "sao_paulo"]
# Beijing (88 scored test queries) is excluded from the paper; set GEMMA_WITH_BEIJING=1 to bring it back.
TGT = ["tokyo", "shanghai", "palembang", "melbourne", "sydney", "tangerang", "moscow", "bandung"]
if os.environ.get("GEMMA_WITH_BEIJING"):
    TGT.insert(1, "beijing")
PRETTY = {"istanbul": "Istanbul", "petaling_jaya": "Petaling Jaya", "jakarta": "Jakarta",
          "new_york": "New York", "kuwait_city": "Kuwait City", "sao_paulo": "São Paulo",
          "tokyo": "Tokyo", "beijing": "Beijing", "shanghai": "Shanghai", "palembang": "Palembang",
          "melbourne": "Melbourne", "sydney": "Sydney", "tangerang": "Tangerang", "moscow": "Moscow",
          "bandung": "Bandung"}
# Okabe-Ito based palette for the ~11 Foursquare top-level activities (colour-blind safe)
ACT_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
              "#999999", "#000000", "#882255", "#44AA99", "#AA4499"]


def pretty(c):
    return PRETTY.get(c, c)


# short labels for Foursquare top-level activities (figure legends / annotations)
SHORT_ACT = {"Arts and Entertainment": "Arts & Ent.", "Business and Professional Services": "Work/Business",
             "Community and Government": "Community", "Dining and Drinking": "Dining",
             "Health and Medicine": "Health", "Landmarks and Outdoors": "Outdoors", "Retail": "Shopping",
             "Sports and Recreation": "Sports", "Travel and Transportation": "Transit", "Event": "Event"}


def short_act(name):
    name = str(name)
    return SHORT_ACT.get(name, name.split(" and ")[0][:14])


def base_args(desc, cities_default=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--run_dir", default="runs/gemma_v4", help="run dir whose config.json gives data paths")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--fsq_csv", default=None)
    ap.add_argument("--cities", nargs="+", default=cities_default or (SRC + TGT))
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def load_cfg(a):
    from gemma.downstream import load_config
    cfg = load_config(a.run_dir)
    if a.data_root: cfg.data_root = a.data_root
    if a.fsq_csv: cfg.fsq_csv = a.fsq_csv
    return cfg


def taxonomy(cfg):
    from gemma.fsq import FSQTaxonomy
    fsq = FSQTaxonomy(cfg.fsq_csv)
    names = {v: k for k, v in fsq.act2id.items()}
    return fsq, names


def load_city(cfg, fsq, city):
    """Light loader -> dict with a check-in frame and trajectories (same rules as gemma.data)."""
    from gemma.data import load_city_csv
    df = load_city_csv(cfg.data_root, city, ["train", "validation", "test"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    if "venue_city_latitude" in df.columns:
        df["lat"] = df["lat"].fillna(pd.to_numeric(df["venue_city_latitude"], errors="coerce"))
        df["lon"] = df["lon"].fillna(pd.to_numeric(df["venue_city_longitude"], errors="coerce"))
    df["act"] = [fsq.activity_id(c) for c in df["venue_category_id"]]
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["dow"] = df["timestamp"].dt.dayofweek
    trails = []
    for (uid, tid), g in df.groupby(["user_id", "trail_id"], sort=False):
        rows = g.index.to_numpy()
        if len(rows) < cfg.min_trail_len:
            continue
        trails.append({"user_id": uid, "trail_id": tid, "rows": rows, "split": g["_split"].iloc[0],
                       "t0": g["timestamp"].iloc[0]})
    return {"city": city, "df": df, "trails": trails}


def sthgcn_test_queries(C):
    """(trail, target_row) for the STHGCN protocol: last check-in of each test trajectory whose
    POI and user occur in the train split; history = the user's train trajectories."""
    df = C["df"]
    tr = [t for t in C["trails"] if t["split"] == "train"]
    train_pois = set(df.loc[np.concatenate([t["rows"] for t in tr]), "venue_id"].tolist()) if tr else set()
    train_users = {t["user_id"] for t in tr}
    out = []
    for t in C["trails"]:
        if t["split"] != "test" or len(t["rows"]) < 2:
            continue
        r = t["rows"][-1]
        if df.at[r, "venue_id"] in train_pois and t["user_id"] in train_users:
            out.append((t, r))
    return out, tr


def user_history(train_trails):
    h = defaultdict(list)
    for t in train_trails:
        h[t["user_id"]].append(t)
    return h


def outdirs(out):
    for d in ("tables", "figures"):
        os.makedirs(os.path.join(out, d), exist_ok=True)


def save_table(df, out, name, caption="", float_fmt="{:.3f}"):
    outdirs(out)
    df.to_csv(os.path.join(out, "tables", name + ".csv"), index=False)
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
             "\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(str(c).replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else float_fmt.format(v))
            else:
                cells.append(str(v).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(out, "tables", name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] {name}\n{df.to_string(index=False)}\n")


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    if os.environ.get("GEMMA_PAPER_STYLE"):     # same fonts/sizes/colours as paper/figures.py
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper"))
        from style import setup
        setup()
    return plt


def save_fig(fig, out, name):
    outdirs(out)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), bbox_inches="tight")
    print(f"[figure] {os.path.join(out, 'figures', name)}.pdf/.png")


def city_style(c):
    """sources dashed grey-ish, targets solid colour."""
    return {"ls": "--" if c in SRC else "-", "lw": 1.0 if c in SRC else 1.4}


def js_div(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps; q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def dump_json(obj, out, name):
    outdirs(out)
    json.dump(obj, open(os.path.join(out, "tables", name + ".json"), "w"), indent=2, default=float)
