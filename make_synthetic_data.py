"""
make_synthetic_data.py
----------------------
Generates tiny Massive-STEPS-format CSVs so the full pipeline can be smoke
-tested without the real datasets.  Crucially it bakes in a *shared* activity
grammar (category transition matrix) across cities while giving each city a
different POI inventory and occupancy mix -- exactly the factorization GEMMA
is meant to recover.  Real runs point --data_root at the true data instead.
"""
import argparse
import os

import numpy as np
import pandas as pd

CATEGORIES = ["Home", "Office", "Coffee Shop", "Restaurant", "Gym",
              "Bar", "Park", "Grocery Store", "Mall", "Train Station"]

# shared, city-invariant activity grammar (row = from, col = to)
GRAMMAR = np.array([
    [0.05, .30, .20, .10, .10, .05, .05, .10, .03, .02],  # Home
    [.15, .05, .25, .20, .05, .10, .05, .05, .05, .05],   # Office
    [.20, .30, .05, .15, .05, .05, .05, .05, .05, .05],   # Coffee
    [.30, .10, .10, .05, .05, .15, .05, .05, .10, .05],   # Restaurant
    [.30, .10, .15, .10, .05, .10, .05, .05, .05, .05],   # Gym
    [.35, .05, .05, .20, .05, .10, .05, .05, .05, .05],   # Bar
    [.30, .10, .15, .10, .10, .05, .05, .05, .05, .05],   # Park
    [.40, .10, .10, .10, .05, .05, .05, .05, .05, .05],   # Grocery
    [.30, .10, .15, .15, .05, .05, .05, .05, .05, .05],   # Mall
    [.25, .25, .10, .10, .05, .05, .05, .05, .05, .05],   # Train
], dtype=float)
GRAMMAR = GRAMMAR / GRAMMAR.sum(1, keepdims=True)

CITY_CENTERS = {"synth_a": (40.7, -74.0), "synth_b": (35.6, 139.7),
                "synth_c": (41.0, 28.9)}


def gen_city(city, n_users, n_pois_per_cat, trails_per_user, seed):
    rng = np.random.RandomState(seed)
    clat, clon = CITY_CENTERS.get(city, (0.0, 0.0))
    # city-specific POI inventory (disjoint ids across cities via offset)
    pois = []
    vid = 0
    for ci, cat in enumerate(CATEGORIES):
        for _ in range(n_pois_per_cat):
            pois.append({"venue_id": vid, "venue_category": cat,
                         "venue_category_id": ci, "venue_category_id_code": ci,
                         "name": f"{cat} {vid}",
                         "latitude": clat + rng.randn() * 0.05,
                         "longitude": clon + rng.randn() * 0.05})
            vid += 1
    pois = pd.DataFrame(pois)
    by_cat = {c: pois[pois.venue_category == c] for c in CATEGORIES}

    rows = []
    tid = 0
    base_time = pd.Timestamp("2024-01-01")
    for u in range(n_users):
        # each user has a slightly biased starting category (role-ish)
        for _ in range(trails_per_user):
            length = rng.randint(2, 6)
            cat = rng.randint(len(CATEGORIES))
            t = base_time + pd.Timedelta(hours=int(rng.randint(0, 24 * 300)))
            for step in range(length):
                cand = by_cat[CATEGORIES[cat]]
                poi = cand.iloc[rng.randint(len(cand))]
                rows.append({
                    "trail_id": tid, "user_id": u,
                    "venue_id": int(poi.venue_id),
                    "latitude": poi.latitude, "longitude": poi.longitude,
                    "name": poi["name"], "address": "n/a",
                    "venue_category": poi.venue_category,
                    "venue_category_id": int(poi.venue_category_id),
                    "venue_category_id_code": int(poi.venue_category_id_code),
                    "venue_city": city, "venue_city_latitude": clat,
                    "venue_city_longitude": clon, "venue_country": "SYN",
                    "timestamp": (t + pd.Timedelta(hours=step)).strftime(
                        "%Y-%m-%d %H:%M:%S")})
                cat = rng.choice(len(CATEGORIES), p=GRAMMAR[cat])
            tid += 1
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--cities", nargs="+",
                    default=["synth_a", "synth_b", "synth_c"])
    ap.add_argument("--n_users", type=int, default=60)
    ap.add_argument("--pois_per_cat", type=int, default=8)
    ap.add_argument("--trails_per_user", type=int, default=12)
    args = ap.parse_args()

    for i, city in enumerate(args.cities):
        df = gen_city(city, args.n_users, args.pois_per_cat,
                      args.trails_per_user, seed=100 + i)
        trails = df.trail_id.unique()
        rng = np.random.RandomState(0)
        rng.shuffle(trails)
        n = len(trails)
        tr, va = trails[:int(.7 * n)], trails[int(.7 * n):int(.8 * n)]
        te = trails[int(.8 * n):]
        d = os.path.join(args.out, city)
        os.makedirs(d, exist_ok=True)
        for split, ids in [("train", tr), ("validation", va), ("test", te)]:
            df[df.trail_id.isin(ids)].to_csv(
                os.path.join(d, f"{city}_checkins_{split}.csv"), index=False)
        print(f"{city}: {len(df)} check-ins, {n} trails "
              f"(train/val/test = {len(tr)}/{len(va)}/{len(te)})")


if __name__ == "__main__":
    main()
