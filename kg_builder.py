#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, math
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ---------- helpers ----------
def ent_poi(vid):     return f"POI::{str(vid)}"
def ent_cat(cat):     return f"CAT::{str(cat)}"
def ent_city(c):      return f"CITY::{str(c)}"
def ent_hour(h):      return f"HOUR::{int(h)}"
def ent_wday(w):      return f"WDAY::{int(w)}"
def ent_grid(gx,gy):  return f"GRID::{gx}:{gy}"

def grid_xy(lat, lon, cell_deg=0.01):
    # ~1.1km at equator (crude but fast)
    gx = int(math.floor(float(lat)/cell_deg))
    gy = int(math.floor(float(lon)/cell_deg))
    return gx, gy

def safe_grid_xy(r, cell_deg_val):
    lat, lon = r["latitude"], r["longitude"]
    if pd.isna(lat) or pd.isna(lon):
        return (np.nan, np.nan)
    try:
        return grid_xy(lat, lon, cell_deg=cell_deg_val)
    except Exception:
        return (np.nan, np.nan)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    import math
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_city_df(data_root, city):
    # expects: <city>/<city>_checkins_{train,validation,test}.csv
    def _load(split):
        p = os.path.join(data_root, city, f"{city}_checkins_{split}.csv")
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return _load("train"), _load("validation"), _load("test")

# ---------- triple mining ----------
def build_triples_from_city(df_train, df_val, df_test,
                            delta_hours=8,
                            proximity_km=0.2,
                            cell_deg=0.01):
    """
    Use only train+val to construct KG (no test leakage).
    """
    base = pd.concat([df_train, df_val], ignore_index=True)
    base = base.dropna(subset=["venue_id"])
    base["venue_id"] = base["venue_id"].astype(str)
    base["timestamp"] = pd.to_datetime(base["timestamp"])

    # Deduplicate POI metadata (latest)
    poi_meta = (base.sort_values("timestamp")
                    .drop_duplicates("venue_id", keep="last"))

    triples = defaultdict(int)  # (h,r,t) -> weight count

    # 1) Hierarchy edges for each POI
    for r in tqdm(poi_meta.itertuples(index=False), total=len(poi_meta), desc="Hierarchy"):
        vid   = getattr(r, "venue_id")
        name  = str(getattr(r, "name", "") or "")
        lat   = float(getattr(r, "latitude"))
        lon   = float(getattr(r, "longitude"))
        city  = str(getattr(r, "venue_city", "") or "")
        cat   = str(getattr(r, "venue_category", "") or "")
        p     = ent_poi(vid)

        if cat:
            triples[(p, "belongs_to_category", ent_cat(cat))] += 1
        if city:
            triples[(p, "located_in_city", ent_city(city))] += 1

        if not (math.isnan(lat) or math.isnan(lon)):
            gx, gy = grid_xy(lat, lon, cell_deg=cell_deg)
            triples[(p, "in_spatial_grid", ent_grid(gx, gy))] += 1
        else:
            # Skip if coordinates missing
            continue


    # 2) Temporal activity edges by hour / weekday
    base["hour"] = base["timestamp"].dt.hour
    base["wday"] = base["timestamp"].dt.weekday

    for (vid, h), g in tqdm(base.groupby(["venue_id","hour"]), desc="Hour activity"):
        triples[(ent_poi(vid), "active_in_hour", ent_hour(int(h)))] += len(g)

    for (vid, w), g in tqdm(base.groupby(["venue_id","wday"]), desc="Weekday activity"):
        triples[(ent_poi(vid), "active_on_wday", ent_wday(int(w)))] += len(g)

    # 3) Proximity edges between POIs in same grid cell (symmetric)
    # Build index by grid
    poi_positions = poi_meta[["venue_id","latitude","longitude"]].copy()
    #poi_positions["gx"], poi_positions["gy"] = zip(*poi_positions.apply(lambda r: grid_xy(r["latitude"], r["longitude"], cell_deg), axis=1))
    poi_positions["gx"], poi_positions["gy"] = zip(
        *poi_positions.apply(lambda row: safe_grid_xy(row, cell_deg), axis=1)
    )
    poi_positions = poi_positions.dropna(subset=["gx", "gy"]).reset_index(drop=True)
    grid_index = defaultdict(list)
    for row in poi_positions.itertuples(index=False):
        grid_index[(row.gx, row.gy)].append((row.venue_id, float(row.latitude), float(row.longitude)))

    for key, lst in tqdm(grid_index.items(), desc="Proximity"):
        n = len(lst)
        for i in range(n):
            vi, la, lo = lst[i]
            for j in range(i+1, n):
                vj, lb, lblo = lst[j][0], lst[j][1], lst[j][2]
                d = haversine_km(la, lo, lb, lblo)
                if d <= proximity_km:
                    triples[(ent_poi(vi), "near", ent_poi(vj))] += 1
                    triples[(ent_poi(vj), "near", ent_poi(vi))] += 1

    # 4) Co-visit / transition edges within δτ = 8h per user
    delta = pd.Timedelta(hours=delta_hours)
    for uid, g in tqdm(base.sort_values("timestamp").groupby("user_id"), desc="Transitions"):
        seq = list(g[["venue_id","timestamp"]].itertuples(index=False, name=None))
        for i in range(1, len(seq)):
            v_prev, t_prev = seq[i-1]
            v_next, t_next = seq[i]
            if (t_next - t_prev) <= delta and v_prev != v_next:
                triples[(ent_poi(v_prev), "co_visit_next", ent_poi(v_next))] += 1

    return triples

def compile_kg(triples_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Materialize triples with weights (repeat by count or store weight attribute)
    # For KGE, simple repetition is standard; here we expand lightly (cap).
    rows = []
    for (h,r,t), w in triples_dict.items():
        reps = min(w, 10)  # cap repetition to avoid huge files
        rows.extend([(h,r,t)] * reps)

    # mappings
    ents = sorted(set([h for h,_,_ in rows] + [t for _,_,t in rows]))
    rels = sorted(set([r for _,r,_ in rows]))
    entity2id = {e:i for i,e in enumerate(ents)}
    rel2id    = {r:i for i,r in enumerate(rels)}

    # save
    with open(os.path.join(out_dir, "triples.tsv"), "w") as f:
        for h,r,t in rows:
            f.write(f"{h}\t{r}\t{t}\n")
    json.dump(entity2id, open(os.path.join(out_dir, "entity2id.json"), "w"))
    json.dump(rel2id,    open(os.path.join(out_dir, "rel2id.json"), "w"))

    print(f"[KG] {len(rows)} triples | {len(entity2id)} entities | {len(rel2id)} relations")
    print(f"[KG] saved to: {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with per-city subfolders")
    ap.add_argument("--city", required=True, help="e.g., new_york")
    ap.add_argument("--out",   required=True, help="Output KG folder, e.g., runs/new_york/kg")
    ap.add_argument("--delta_hours", type=int, default=8)
    ap.add_argument("--cell_deg", type=float, default=0.01)
    ap.add_argument("--proximity_km", type=float, default=0.2)
    args = ap.parse_args()

    tr, va, te = load_city_df(args.data_root, args.city)
    triples = build_triples_from_city(tr, va, te,
                                      delta_hours=args.delta_hours,
                                      proximity_km=args.proximity_km,
                                      cell_deg=args.cell_deg)
    compile_kg(triples, args.out)

if __name__ == "__main__":
    main()
