"""gemma.collab -- population (collaborative) statistics of a city's check-in log for GEMMA v7.

Why: STHGCN's advantage in data-rich cities is collaborative signal (other users' trajectories,
encoded in its hypergraph).  GEMMA v6 used the city log only as raw counts in the EXPLORE channel,
with hand-set weights that were never learned on the source cities (source training ran with
stat=None).  v7 adds three scale-free population signals to BOTH channels and learns their
weights on the source cities (leave-one-out), so they transfer like the personal evidence does.

  ctr(v)   smoothed population transition   log P(v | last) with hierarchical back-off to the
           category transition:  (c(last->v) + k * Pcat(v|last)) / (out(last) + k)
           Pcat(v|last) = T[sec(last), sec(v)] / n_sec(sec(v))   (sums to 1 over POIs)
  ccat(v)  log T[sec(last), sec(v)]   category-level transition of the whole city
  icf(v)   item-item collaborative filtering: mean cosine similarity (co-visiting users)
           between v and the places in the user's history, recency weighted.
           User-level leave-one-out (the query user's own co-visits are removed) at train
           AND test time, so the feature means the same thing in both.

Leakage control: transition counts use the existing leave-one-trajectory-out correction of
realize.CityIndex (c_cov) plus the trajectory's own out-degree (meta 'own_out'); icf removes
the user's own contribution; the category matrix aggregates thousands of transitions (a
single trajectory changes it negligibly).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    import scipy.sparse as sp
except Exception:                      # pragma: no cover
    sp = None


class CollabStats:
    def __init__(self, ci, stat_ids, user_cap=30, topk=100, logger=None):
        if sp is None:
            raise ImportError("scipy is required for GEMMA v7 collaborative statistics")
        P = ci.P
        sec = ci.sec.cpu().numpy()
        S = int(sec.max()) + 1
        self.P, self.S = P, S
        self.sec = sec
        self.n_sec = np.bincount(sec, minlength=S).astype(np.float32)
        # ---- category transitions + per-POI out-degree
        T = np.zeros((S, S), np.float64)
        out = np.zeros(P, np.float32)
        per_user = defaultdict(lambda: defaultdict(float))
        for i in stat_ids:
            tr = ci.trails[i]; pis = ci.row_pi[tr["rows"]]
            if len(pis) > 1:
                np.add.at(T, (sec[pis[:-1]], sec[pis[1:]]), 1.0)
                np.add.at(out, pis[:-1], 1.0)
            u = tr["user_id"]
            for v in pis:
                per_user[u][int(v)] += 1.0
        T = (T + 0.1) / (T.sum(1, keepdims=True) + 0.1 * S)
        self.logT = np.log(T).astype(np.float32)                 # (S,S)
        self.out = out
        # ---- item-item cosine over users (top user_cap places per user)
        rows, cols = [], []
        self.user_set = {}
        for k, (u, d) in enumerate(per_user.items()):
            top = [v for v, _ in sorted(d.items(), key=lambda x: -x[1])[:user_cap]]
            self.user_set[u] = set(top)
            rows.extend([k] * len(top)); cols.extend(top)
        X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(max(1, len(per_user)), P))
        deg = np.asarray(X.sum(0)).ravel().astype(np.float32)
        C = (X.T @ X).tocsr()
        C.setdiag(0); C.eliminate_zeros()
        inv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
        C = sp.diags(inv) @ C @ sp.diags(inv)                    # cosine similarity
        C = C.tocsr()
        self.C = self._topk(C, topk)
        self.deg = deg
        self.inv = inv
        if logger:
            logger.info(f"[collab] users={len(per_user)} item-item nnz={self.C.nnz} "
                        f"category transitions={S}x{S} (top-{topk}, cap {user_cap}/user)")

    @staticmethod
    def _topk(C, k):
        C = C.tocsr()
        indptr, indices, data = [0], [], []
        for r in range(C.shape[0]):
            a, b = C.indptr[r], C.indptr[r + 1]
            if b - a > k:
                sel = np.argpartition(-C.data[a:b], k)[:k]
                indices.append(C.indices[a:b][sel]); data.append(C.data[a:b][sel])
            else:
                indices.append(C.indices[a:b]); data.append(C.data[a:b])
            indptr.append(indptr[-1] + len(indices[-1]))
        return sp.csr_matrix((np.concatenate(data) if data else np.zeros(0, np.float32),
                              np.concatenate(indices) if indices else np.zeros(0, np.int32),
                              np.asarray(indptr)), shape=C.shape)

    def icf(self, hw, user):
        """hw: {poi: weight} of the user's history -> (pois, values) of log1p(20*mean cosine)."""
        if not hw:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        idx = np.fromiter(hw.keys(), np.int64); w = np.fromiter(hw.values(), np.float32)
        w = w / max(w.sum(), 1e-9)
        row = sp.csr_matrix((w, (np.zeros(len(idx), np.int64), idx)), shape=(1, self.P)) @ self.C
        row = row.tocsr()
        pois, vals = row.indices.astype(np.int64), row.data.astype(np.float32).copy()
        U = self.user_set.get(user)
        if U:      # remove the user's own co-visits (user-level leave-one-out)
            wmap = dict(zip(idx.tolist(), w.tolist()))
            base = sum(wmap.get(h, 0.0) * self.inv[h] for h in U)
            for k, v in enumerate(pois.tolist()):
                if v in U:
                    own = base - wmap.get(v, 0.0) * self.inv[v]
                    vals[k] = max(0.0, vals[k] - own * self.inv[v])
        keep = vals > 0
        return pois[keep], np.log1p(20.0 * vals[keep]).astype(np.float32)
