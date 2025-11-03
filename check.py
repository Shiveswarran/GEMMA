import json, torch, pandas as pd

# Load mappings and sizes
poi_index = json.load(open("runs/new_york/name_emb/poi_index.json"))
kg_raw = torch.load("runs/new_york/kg/entity_embeddings.pt", map_location="cpu")
print("POIs:", len(poi_index))
print("KG entities:", kg_raw.shape[0])

# Load one dataset to test
df = pd.read_csv("data/new_york/new_york_checkins_train.csv")
missing = []
for vid in df["venue_id"].astype(str).unique():
    if vid not in poi_index:
        missing.append(vid)
print("Missing from poi_index:", len(missing))

# Check mapping coverage between name_emb and KGE
entity2id = json.load(open("runs/new_york/kg/entity2id.json"))
cnt = 0
for vid in poi_index:
    if f"POI::{vid}" in entity2id:
        cnt += 1
print(f"KGE coverage: {cnt}/{len(poi_index)} ({100*cnt/len(poi_index):.2f}%)")
