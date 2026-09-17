"""Offline: precompute + cache modern-LLM POI-knowledge embeddings for poi_encoder=llm_cached.
Swap MODEL for any encoder (GTE-Qwen, OpenAI text-embedding, ...). Keyed by md5(text)."""
import argparse, glob, hashlib, os, sys
import numpy as np, pandas as pd
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",default="data"); ap.add_argument("--cities",nargs="+",required=True)
    ap.add_argument("--cache",default="poi_emb_cache")
    ap.add_argument("--model",default="sentence-transformers/all-MiniLM-L6-v2")
    a=ap.parse_args(); os.makedirs(a.cache,exist_ok=True)
    texts=set()
    for c in a.cities:
        for f in glob.glob(os.path.join(a.data_root,c,f"{c}_checkins_*.csv")):
            df=pd.read_csv(f, usecols=["venue_category","name"])
            for _,r in df.iterrows():
                texts.add(f"{r['venue_category']} : {r.get('name','')}")
    texts=sorted(texts)
    print(f"encoding {len(texts)} unique POI texts with {a.model}")
    from sentence_transformers import SentenceTransformer   # replace with your LLM encoder
    import torch
    m=SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
    vecs=m.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    keys=[hashlib.md5(t.encode()).hexdigest() for t in texts]
    np.savez(os.path.join(a.cache,"llm_emb.npz"), keys=np.array(keys), vecs=np.asarray(vecs,np.float32))
    print("saved ->", os.path.join(a.cache,"llm_emb.npz"))
if __name__=="__main__": main()
