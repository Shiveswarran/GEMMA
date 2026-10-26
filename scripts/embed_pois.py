"""Precompute the sentence embedding of every place text "<category> : <name>".

python scripts/embed_pois.py --data_root data --cities istanbul ... bandung
-> poi_emb_cache/llm_emb.npz  (keys = md5(text), vecs = L2-normalized embeddings)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.text_encoder import text_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--cache", default="poi_emb_cache")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    texts = set()
    for c in a.cities:
        for f in glob.glob(os.path.join(a.data_root, c, f"{c}_checkins_*.csv")):
            df = pd.read_csv(f, usecols=["venue_category", "name"])
            texts.update(f"{r['venue_category']} : {r.get('name', '')}" for _, r in df.iterrows())
    texts = sorted(texts)
    print(f"encoding {len(texts)} place texts with {a.model}")
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
    vecs = m.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    np.savez(os.path.join(a.cache, "llm_emb.npz"), keys=np.array([text_key(t) for t in texts]),
             vecs=np.asarray(vecs, np.float32))
    print("saved ->", os.path.join(a.cache, "llm_emb.npz"))


if __name__ == "__main__":
    main()
