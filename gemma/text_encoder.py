"""Language embedding of a place's text ("<category> : <name>").

``cached`` looks up sentence embeddings precomputed by ``scripts/embed_pois.py``
(all-MiniLM-L6-v2 by default), keyed by the md5 of the text.  ``hash`` is a deterministic
bag-of-words hashing encoder that needs no download; it is used by the tests and as the
fallback for a text missing from the cache.  All vectors are L2-normalized.
"""
from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np


def text_key(text: str) -> str:
    return hashlib.md5(str(text).encode()).hexdigest()


class TextEncoder:
    def __init__(self, cfg, logger=None):
        self.dim = cfg.poi_emb_dim
        self.backend = "cached" if cfg.poi_encoder in ("cached", "llm_cached") else "hash"
        self._store = {}
        if self.backend == "cached":
            path = os.path.join(cfg.poi_emb_cache, "llm_emb.npz")
            if os.path.exists(path):
                z = np.load(path, allow_pickle=True)
                self._store = dict(zip(z["keys"], z["vecs"]))
                self.dim = z["vecs"].shape[1]
                if logger:
                    logger.info(f"loaded {len(self._store)} cached place embeddings (dim {self.dim})")
            elif logger:
                logger.warning(f"no embedding cache at {path}; falling back to the hash encoder "
                               f"(run scripts/embed_pois.py)")

    def _hash_vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in str(text).lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._store:
            out = np.zeros((len(texts), self.dim), np.float32)
            for i, t in enumerate(texts):
                out[i] = self._store.get(text_key(t), self._hash_vec(t))
            return out
        return np.stack([self._hash_vec(t) for t in texts]).astype(np.float32)
