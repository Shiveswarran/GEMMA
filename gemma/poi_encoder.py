"""
gemma.poi_encoder
-----------------
Pluggable POI-knowledge encoder over POI text (category + name).  Swappable
so the "BERT is old" concern is a one-line config change:

  hash        : deterministic offline hashing (no network) -- for smoke tests.
  sbert       : sentence-transformers (all-MiniLM) -- baseline.
  llm_cached  : load precomputed LLM embeddings (e.g. from an instruction-tuned
                encoder / GTE-Qwen / OpenAI text-embedding) keyed by text hash.
                Compute them offline once (see scripts/embed_pois.py contract in
                README) and drop the .npy + keys here; this class just looks up.

All encoders return L2-normalized vectors of dim `poi_emb_dim`, disk-cached.
"""
from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np


class POIEncoder:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.dim = cfg.poi_emb_dim
        self.backend = cfg.poi_encoder
        self.logger = logger
        self._model = None
        os.makedirs(cfg.poi_emb_cache, exist_ok=True)
        if self.backend == "sbert":
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(cfg.sbert_model, device=dev)
                self.dim = self._model.get_sentence_embedding_dimension()
                if logger:
                    logger.info(f"SBERT {cfg.sbert_model} dim={self.dim} dev={dev}")
            except Exception as e:      # graceful fallback (offline sandboxes)
                if logger:
                    logger.warning(f"SBERT unavailable ({e}); using hash encoder")
                self.backend = "hash"
        if self.backend == "llm_cached":
            self._store = {}
            path = os.path.join(cfg.poi_emb_cache, "llm_emb.npz")
            if os.path.exists(path):
                z = np.load(path, allow_pickle=True)
                self._store = {k: v for k, v in zip(z["keys"], z["vecs"])}
                self.dim = z["vecs"].shape[1]
                if logger:
                    logger.info(f"loaded {len(self._store)} cached LLM embeddings")
            elif logger:
                logger.warning("llm_cached selected but no cache found; hashing")

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in str(text).lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sbert" and self._model is not None:
            v = self._model.encode(texts, batch_size=256, show_progress_bar=False,
                                   normalize_embeddings=True)
            return np.asarray(v, np.float32)
        if self.backend == "llm_cached" and getattr(self, "_store", None):
            out = np.zeros((len(texts), self.dim), np.float32)
            for i, t in enumerate(texts):
                key = hashlib.md5(str(t).encode()).hexdigest()
                out[i] = self._store.get(key, self._hash_vec(t))
            return out
        return np.stack([self._hash_vec(t) for t in texts]).astype(np.float32)
