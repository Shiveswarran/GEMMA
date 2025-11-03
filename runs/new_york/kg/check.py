import torch, json
kg = torch.load("/scratch/umni5/a/shives/POI_RecSys/Multi-Tower-Embedding/runs/new_york/kg/entity_embeddings.pt", map_location="cpu")
print(type(kg))
if isinstance(kg, dict):
    print("keys:", kg.keys())
    kg = kg.get("emb", next(iter(kg.values())))
print("Shape:", getattr(kg, "shape", None))
e2id = json.load(open("/scratch/umni5/a/shives/POI_RecSys/Multi-Tower-Embedding/runs/new_york/kg/entity2id.json"))
print("Entities in mapping:", len(e2id))
