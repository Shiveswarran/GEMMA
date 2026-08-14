"""
gemma.features
--------------
Turns the raw Massive-STEPS columns into the *city-agnostic* numeric streams
that make up a GEMMA check-in token:

  * POI-name  -> frozen text embedding (SBERT, cached to disk).  An offline
                 hashing embedder is provided so the pipeline runs even when
                 the SBERT weights cannot be downloaded (e.g. in a sandbox).
  * category  -> shared global category id (+ optional frozen SBERT of the
                 category string, since the category set B is shared across
                 cities and is the transferable label space).
  * time      -> cyclic sinusoidal encodings of time-of-day / day-of-week /
                 month(season), plus a log dwell-time feature.
  * space     -> RELATIVE motion features (distance & bearing to the previous
                 check-in).  Absolute coordinates are deliberately NOT part of
                 the invariant token (identifier / coordinate shift); city and
                 country are handled separately in the realization map.

Everything here is deterministic and cache-friendly so repeated runs are fast.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
#  Text embedding (POI names and category strings)                            #
# --------------------------------------------------------------------------- #
class NameEmbedder:
    """Frozen text embedder with an on-disk cache.

    backend="sbert": uses sentence-transformers (the intended production path).
    backend="hash" : deterministic character-n-gram hashing embedder that needs
                     no network / weights.  Used as an automatic fallback and
                     for smoke tests; results are stable across runs.
    """

    def __init__(self, backend: str = "sbert", model_name: str = "",
                 dim: int = 384, cache_dir: str = "cache/text", logger=None):
        self.backend = backend
        self.model_name = model_name
        self.dim = dim
        self.cache_dir = cache_dir
        self.logger = logger
        os.makedirs(cache_dir, exist_ok=True)
        self._mem: Dict[str, np.ndarray] = {}
        self._model = None

        if backend == "sbert":
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(model_name, device=dev)
                self.dim = self._model.get_sentence_embedding_dimension()
                if logger:
                    logger.info(f"SBERT loaded: {model_name} "
                                f"(dim={self.dim}, device={dev})")
            except Exception as e:  # pragma: no cover - depends on environment
                if logger:
                    logger.warning(
                        f"Could not load SBERT ({e!r}); falling back to hash "
                        f"embedder (dim={dim}). Install sentence-transformers and "
                        f"re-run for real name semantics.")
                self.backend = "hash"

    # -- hashing fallback ---------------------------------------------------
    def _hash_vec(self, text: str) -> np.ndarray:
        text = (text or "").lower().strip()
        v = np.zeros(self.dim, dtype=np.float32)
        toks = text.split() + [text[i:i + 3] for i in range(max(0, len(text) - 2))]
        if not toks:
            toks = ["<empty>"]
        for tok in toks:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 32) & 1 else -1.0
            v[idx] += sign
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # -- public API ---------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of strings -> (N, dim) float32, with per-string cache."""
        texts = [str(t) if t is not None else "" for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing_idx, missing_txt = [], []
        for i, t in enumerate(texts):
            key = self._key(t)
            if key in self._mem:
                out[i] = self._mem[key]
                continue
            path = os.path.join(self.cache_dir, key + ".npy")
            if os.path.exists(path):
                vec = np.load(path)
                self._mem[key] = vec
                out[i] = vec
            else:
                missing_idx.append(i)
                missing_txt.append(t)

        if missing_txt:
            if self.backend == "sbert":
                vecs = self._model.encode(
                    missing_txt, batch_size=256, show_progress_bar=False,
                    normalize_embeddings=True).astype(np.float32)
            else:
                vecs = np.stack([self._hash_vec(t) for t in missing_txt])
            for j, i in enumerate(missing_idx):
                key = self._key(missing_txt[j])
                self._mem[key] = vecs[j]
                np.save(os.path.join(self.cache_dir, key + ".npy"), vecs[j])
                out[i] = vecs[j]
        return out

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
#  Category vocabulary (shared across cities = transferable label space)      #
# --------------------------------------------------------------------------- #
class CategoryVocab:
    """Maps the shared Foursquare-style category strings to global ids.

    Massive-STEPS uses a shared category taxonomy across cities, so a single
    global vocabulary built over *all* cities is what makes cross-city grammar
    transfer possible.  We also expose a coarse "primary" category (first token
    before a common separator) to approximate the hierarchical (main->secondary)
    structure the paper describes when an explicit hierarchy is not present.
    """

    SEP_TOKENS = [" > ", " / ", ":", " - "]

    def __init__(self):
        self.cat2id: Dict[str, int] = {"<unk>": 0}
        self.primary2id: Dict[str, int] = {"<unk>": 0}
        self.id2cat: List[str] = ["<unk>"]
        self.use_coarse: bool = False
        self.cat2coarse: Dict[str, int] = {}
        self._n_coarse: int = 0

    def build_coarse(self, embedder, n_coarse: int, logger=None) -> "CategoryVocab":
        """Induce a COARSE, city-invariant activity label space by clustering
        category-name embeddings.  Massive-STEPS categories are flat leaf labels
        (no hierarchy separator), so a semantic clustering is what yields the
        ~10-20 shared "activities" the factorization premise needs."""
        from sklearn.cluster import KMeans
        cats = [c for c in self.id2cat if c != "<unk>"]
        if len(cats) <= n_coarse:                    # already coarse enough
            self.cat2coarse = {c: i + 1 for i, c in enumerate(cats)}
            self._n_coarse = len(cats) + 1
            self.use_coarse = True
            return self
        emb = embedder.encode(cats)
        km = KMeans(n_clusters=n_coarse, random_state=42, n_init=10).fit(emb)
        self.cat2coarse = {c: int(lbl) + 1 for c, lbl in zip(cats, km.labels_)}
        self._n_coarse = n_coarse + 1                 # +1 for <unk> = 0
        self.use_coarse = True
        if logger:
            logger.info(f"coarse activity space: {len(cats)} categories "
                        f"-> {n_coarse} activities (semantic clusters)")
        return self

    def _primary_of(self, cat: str) -> str:
        c = str(cat)
        for sep in self.SEP_TOKENS:
            if sep in c:
                return c.split(sep)[0].strip()
        return c.strip()

    def fit(self, categories: Sequence[str]) -> "CategoryVocab":
        for cat in categories:
            cat = str(cat)
            if cat not in self.cat2id:
                self.cat2id[cat] = len(self.cat2id)
                self.id2cat.append(cat)
            prim = self._primary_of(cat)
            if prim not in self.primary2id:
                self.primary2id[prim] = len(self.primary2id)
        return self

    def cat_id(self, cat: str) -> int:
        return self.cat2id.get(str(cat), 0)

    def primary_id(self, cat: str) -> int:
        if self.use_coarse:
            return self.cat2coarse.get(str(cat), 0)
        return self.primary2id.get(self._primary_of(str(cat)), 0)

    @property
    def n_cat(self) -> int:
        return len(self.cat2id)

    @property
    def n_primary(self) -> int:
        return self._n_coarse if self.use_coarse else len(self.primary2id)


# --------------------------------------------------------------------------- #
#  Cyclic temporal encoding                                                   #
# --------------------------------------------------------------------------- #
def cyclic_encode(value: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Multi-harmonic sinusoidal encoding of a periodic value.

    Returns an array of shape (..., 2*n_harmonics): [sin(1x),cos(1x),sin(2x),...].
    Using several harmonics lets the model represent sharper temporal structure
    than a single sin/cos pair (e.g. bimodal commute peaks).
    """
    value = np.asarray(value, dtype=np.float32)
    feats = []
    for k in range(1, n_harmonics + 1):
        ang = 2.0 * np.pi * k * value / period
        feats.append(np.sin(ang))
        feats.append(np.cos(ang))
    return np.stack(feats, axis=-1)


def temporal_features(hour: np.ndarray, dow: np.ndarray, month: np.ndarray,
                      dwell_log: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Concatenate cyclic encodings for time-of-day, day-of-week, season(month),
    plus a scalar log dwell time.  Shape: (N, 6*n_harmonics + 1)."""
    tod = cyclic_encode(hour, 24.0, n_harmonics)
    wk = cyclic_encode(dow, 7.0, n_harmonics)
    mo = cyclic_encode(month, 12.0, n_harmonics)
    dwell = np.asarray(dwell_log, dtype=np.float32)[..., None]
    return np.concatenate([tod, wk, mo, dwell], axis=-1)


def temporal_dim(n_harmonics: int) -> int:
    return 6 * n_harmonics + 1


# --------------------------------------------------------------------------- #
#  Relative spatial features                                                  #
# --------------------------------------------------------------------------- #
def spatial_dim() -> int:
    # [log(1+dist_km), sin(bearing), cos(bearing), is_first_flag]
    return 4
