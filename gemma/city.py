"""Per-city index: place tensors, user histories, queries and personal evidence.

``CityIndex(city, observed, stat)``

    observed : trajectories whose check-ins are visible as user history
    stat     : trajectories used for population statistics of the city, or None.
               None in the zero-shot setting; the training split in the fine-tuned setting.

For a query (trajectory i, position j) the target is check-in j and the context is the
prefix [0, j) plus the user's other visible trajectories.  ``batch`` builds the encoder input
[hist_1 SEP hist_2 SEP ... SEP prefix] and every statistic of Section 3.3 of the paper from
the user's own visits:

    hist      recency-decayed visit count (0.9 per trajectory)
    prefix    visits in the current trajectory
    is_last   the place of the last check-in
    ptrans    personal transitions  last place -> v
    tod, dow  visits within two hours of the query time / on the same day type
    rec       exp(-days since the last visit / 14)
    geo       -log(1 + km from the last check-in)
    discovery state: distinct places S_u, per-activity first-visit counts, new share of prefix
    activity profile and activity transitions of the user, five most visited places (anchors)

Population statistics (fine-tuned setting only): transitions last -> v over all users,
popularity and the hour-of-day profile of every place.  For training queries they are computed
leave-one-trajectory-out, so a query never sees its own label in a count.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .data import TT_CHECKIN, TT_SEP


class CityIndex:
    def __init__(self, city, observed, stat, cfg, device):
        self.city, self.cfg, self.device = city, cfg, device
        pt = city.poi_table
        self.P = len(pt["venue_ids"])
        self.row_pi = np.array([pt["id2row"][int(v)] for v in city.venue_id], np.int64)
        ts = city.df["timestamp"]
        self.hour_row = ts.dt.hour.values.astype(np.int64)
        self.day_row = ((ts - ts.min()).dt.total_seconds().values / 86400.0).astype(np.float64)
        self.wkend_row = (ts.dt.dayofweek.values >= 5).astype(np.int64)
        self.act_row = np.asarray(city.act_id, np.int64)
        self.n_act = int(max(int(np.max(pt["act"])), int(np.max(self.act_row)))) + 1
        self._act_groups = None
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        self.lat, self.lon = t(pt["lat"]), t(pt["lon"])
        self.act, self.pri, self.sec = (t(pt[k], torch.long) for k in ("act", "pri", "sec"))
        self.name = t(pt["name_emb"])
        self.codes = None
        if city.codes is not None:
            v2c = {}
            for i in range(len(city.venue_id)):
                v2c.setdefault(int(city.venue_id[i]), city.codes[i])
            c = np.zeros((self.P, city.codes.shape[1]), np.int64)
            for k, v in enumerate(pt["venue_ids"]):
                if int(v) in v2c:
                    c[k] = v2c[int(v)]
            self.codes = t(c, torch.long)

        self.trails = city.trails
        self.observed = set(int(i) for i in observed)
        self.user_trails = defaultdict(list)
        for i in sorted(self.observed, key=lambda i: self.trails[i]["t0"]):
            self.user_trails[self.trails[i]["user_id"]].append(i)
        users = sorted({tr["user_id"] for tr in self.trails})
        self.uid = {u: k for k, u in enumerate(users)}
        self.n_users = len(users)

        self.has_stats = stat is not None
        self.stat = set(int(i) for i in stat) if stat is not None else set()
        if self.has_stats:
            pop = np.zeros(self.P, np.float32)
            H = np.zeros((self.P, 24), np.float32)
            nxt = defaultdict(lambda: defaultdict(float))
            for i in self.stat:
                rows = self.trails[i]["rows"]; pis = self.row_pi[rows]
                np.add.at(pop, pis, 1.0)
                np.add.at(H, (pis, self.hour_row[rows]), 1.0)
                for a, b in zip(pis[:-1], pis[1:]):
                    nxt[int(a)][int(b)] += 1.0
            self.covis = {a: (np.fromiter(d.keys(), np.int64), np.fromiter(d.values(), np.float32))
                          for a, d in nxt.items()}
            self.pop, self.H = t(pop), t(H)
            g = H.sum(0) + 1.0
            self.hprior = t(g / g.sum())

    # ------------------------------------------------------------------ queries
    def history(self, ti):
        """Visible trajectories of the query's user (oldest first), excluding the query."""
        tr = self.trails[ti]
        hs = [j for j in self.user_trails.get(tr["user_id"], []) if j != ti]
        k = self.cfg.hist_last
        if k is not None and k >= 0:          # LLM protocol: the user's last k trajectories
            hs = hs[-k:] if k > 0 else []
        k = self.cfg.hist_budget
        if k is not None and k >= 0:          # causal: the k most recent trajectories before the query
            hs = sorted((j for j in hs if self.trails[j]["t0"] < tr["t0"]),
                        key=lambda j: self.trails[j]["t0"])
            hs = hs[-k:] if k > 0 else []
        return hs

    def queries(self, trail_ids, all_positions):
        """(trajectory, j): predict check-in j from check-ins [0, j)."""
        out = []
        for ti in trail_ids:
            n = len(self.trails[ti]["rows"])
            if n >= 2:
                out += [(ti, j) for j in range(1, n)] if all_positions else [(ti, n - 1)]
        return out

    def act_groups(self):
        if self._act_groups is None:
            a = self.act.cpu().numpy()
            self._act_groups = [torch.as_tensor(np.where(a == k)[0], device=self.device)
                                for k in range(self.n_act)]
        return self._act_groups

    # ------------------------------------------------------------------ batch
    def batch(self, qs, with_target=True):
        """Encoder inputs and sparse personal evidence for a list of queries."""
        cfg, city = self.cfg, self.city
        B, A = len(qs), self.n_act
        toks, meta = [], defaultdict(list)
        sp = {k: ([], [], []) for k in ("hist", "prefix", "ptrans", "c_cov", "c_pop", "c_hour",
                                        "tod", "dow", "rec")}
        u_act, u_atr, u_new = (np.zeros((B, A), np.float32) for _ in range(3))
        u_S, u_pnew = np.zeros(B, np.float32), np.zeros(B, np.float32)
        anchors = np.zeros((B, 5), np.int64)

        def add(key, b, pois, vals):
            sp[key][0].extend([b] * len(pois)); sp[key][1].extend(pois); sp[key][2].extend(vals)

        for b, (ti, j) in enumerate(qs):
            tr = self.trails[ti]; rows = tr["rows"]
            prefix = rows[:j]; last = int(prefix[-1]); last_pi = int(self.row_pi[last])
            hs = self.history(ti)
            # encoder input: [recent history trajectories] SEP ... SEP [prefix]
            tk = []
            for h in hs[-cfg.hist_trails:]:
                tk.extend(self.trails[h]["rows"].tolist()); tk.append(-1)
            tk.extend(prefix.tolist())
            tk = tk[-cfg.hist_max_tokens:]
            while tk and tk[0] == -1:
                tk = tk[1:]
            toks.append(tk)
            # visit counts and personal transitions over the whole visible history
            hs = hs[-200:]
            nh = len(hs)
            for r_i, h in enumerate(hs):
                pis = self.row_pi[self.trails[h]["rows"]]
                add("hist", b, pis.tolist(), [0.9 ** (nh - 1 - r_i)] * len(pis))
                nx = np.where(pis[:-1] == last_pi)[0]
                if len(nx):
                    add("ptrans", b, pis[nx + 1].tolist(), [1.0] * len(nx))
            # time affinity, recency, discovery state, activity profile
            qh, qw, qd = int(self.hour_row[last]), int(self.wkend_row[last]), float(self.day_row[last])
            la_ = int(self.act_row[last])
            seen, cnt, recd = set(), defaultdict(float), {}
            for rws in (self.trails[h]["rows"] for h in hs):
                pis, acts = self.row_pi[rws], self.act_row[rws]
                for k in range(len(rws)):
                    v, a, r = int(pis[k]), int(acts[k]), int(rws[k])
                    u_act[b, a] += 1.0
                    if v not in seen:
                        u_new[b, a] += 1.0; seen.add(v)
                    cnt[v] += 1.0
                    dd = abs(qd - float(self.day_row[r]))
                    if dd < recd.get(v, 1e9):
                        recd[v] = dd
                    hd = abs(int(self.hour_row[r]) - qh); hd = min(hd, 24 - hd)
                    if hd <= 2:
                        add("tod", b, [v], [1.0])
                    if int(self.wkend_row[r]) == qw:
                        add("dow", b, [v], [1.0])
                for k in range(len(rws) - 1):
                    if int(acts[k]) == la_:
                        u_atr[b, int(acts[k + 1])] += 1.0
            for v, dd in recd.items():
                add("rec", b, [v], [float(np.exp(-dd / 14.0))])
            ppis, pacts = self.row_pi[prefix], self.act_row[prefix]
            for k in range(len(prefix) - 1):
                if int(pacts[k]) == la_:
                    u_atr[b, int(pacts[k + 1])] += 1.0
            u_S[b] = float(len(seen))
            u_pnew[b] = float(np.mean([int(v) not in seen for v in ppis])) if len(ppis) else 0.0
            top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
            anchors[b] = [v for v, _ in top] + [last_pi] * (5 - len(top))
            add("prefix", b, ppis.tolist(), [1.0] * len(ppis))
            nx = np.where(ppis[:-1] == last_pi)[0]
            if len(nx):
                add("ptrans", b, ppis[nx + 1].tolist(), [1.0] * len(nx))
            # leave-one-trajectory-out corrections of the population statistics
            if self.has_stats and ti in self.stat:
                apis, hrs = self.row_pi[rows], self.hour_row[rows]
                add("c_pop", b, apis.tolist(), [-1.0] * len(apis))
                msk = hrs == self.hour_row[last]
                add("c_hour", b, apis[msk].tolist(), [-1.0] * int(msk.sum()))
                nx = np.where(apis[:-1] == last_pi)[0]
                add("c_cov", b, apis[nx + 1].tolist(), [-1.0] * len(nx))
            meta["last_pi"].append(last_pi)
            meta["hour"].append(int(self.hour_row[last]))
            meta["user"].append(self.uid[tr["user_id"]])
            if with_target:
                tgt = int(rows[j])
                meta["tgt_pi"].append(int(self.row_pi[tgt]))
                meta["tgt_act"].append(int(city.act_id[tgt]))

        L = max(1, max(len(t) for t in toks))
        R = np.full((B, L), -2, np.int64)
        for b, tk in enumerate(toks):
            R[b, :len(tk)] = tk
        real, sep = R >= 0, R == -1
        Rc = np.clip(R, 0, None)
        tt = lambda x: torch.as_tensor(x, device=self.device)
        enc_in = {"cont": tt((city.cont[Rc] * real[..., None]).astype(np.float32)),
                  "act": tt(np.where(real, city.act_id[Rc], 0)),
                  "pri": tt(np.where(real, city.pri_id[Rc], 0)),
                  "sec": tt(np.where(real, city.sec_id[Rc], 0)),
                  "types": tt(np.where(sep, TT_SEP, TT_CHECKIN).astype(np.int64)),
                  "ctx_mask": tt((real | sep).astype(np.float32))}
        m = {k: tt(np.asarray(v, np.int64)) for k, v in meta.items()}
        m["sparse"] = {k: (tt(np.asarray(v[0], np.int64)), tt(np.asarray(v[1], np.int64)),
                           tt(np.asarray(v[2], np.float32))) for k, v in sp.items()}
        m.update(u_act=tt(u_act), u_atr=tt(u_atr), u_new=tt(u_new), u_S=tt(u_S), u_pnew=tt(u_pnew),
                 anchors=tt(anchors))
        m["wkend"] = tt(np.asarray([int(self.wkend_row[self.trails[ti]["rows"][j - 1]])
                                    for ti, j in qs], np.int64))
        m["tok_pi"] = tt(np.where(real, self.row_pi[Rc], -1).astype(np.int64))
        if with_target and self.codes is not None:
            m["tgt_code"] = self.codes[m["tgt_pi"]]
        m["B"] = B
        return enc_in, m

    # ------------------------------------------------------------------ dense evidence
    def scatter(self, m, key):
        out = torch.zeros(m["B"], self.P, device=self.device)
        b, p, v = m["sparse"][key]
        if b.numel():
            out.index_put_((b, p), v, accumulate=True)
        return out

    def dense(self, m):
        """Dense (B,P) evidence shared by both channels."""
        B, P, dv = m["B"], self.P, self.device
        ar = torch.arange(B, device=dv)
        is_last = torch.zeros(B, P, device=dv)
        is_last[ar, m["last_pi"]] = 1.0
        la, lo = self.lat[m["last_pi"]].unsqueeze(1), self.lon[m["last_pi"]].unsqueeze(1)
        dx = (self.lon.unsqueeze(0) - lo) * 111.32 * torch.cos(torch.deg2rad(la))
        dy = (self.lat.unsqueeze(0) - la) * 111.32
        f = {"hist": torch.log1p(self.scatter(m, "hist")),
             "prefix": torch.log1p(self.scatter(m, "prefix")),
             "ptrans": torch.log1p(self.scatter(m, "ptrans")),
             "is_last": is_last,
             "geo": -torch.log1p(torch.sqrt(dx * dx + dy * dy)),
             "tod": torch.log1p(self.scatter(m, "tod")),
             "dow": torch.log1p(self.scatter(m, "dow")),
             "rec": self._recency(m)}
        zero = torch.zeros(B, P, device=dv)
        f["covis"], f["pop"], f["hour"] = zero, zero, zero
        if self.has_stats:
            cov = torch.zeros(B, P, device=dv)
            bi, pi, vv = [], [], []
            for b, lp in enumerate(m["last_pi"].tolist()):
                if lp in self.covis:
                    ys, cs = self.covis[lp]
                    bi.append(np.full(len(ys), b)); pi.append(ys); vv.append(cs)
            if bi:
                cov.index_put_((torch.as_tensor(np.concatenate(bi), device=dv),
                                torch.as_tensor(np.concatenate(pi), device=dv)),
                               torch.as_tensor(np.concatenate(vv), device=dv), accumulate=True)
            cov = cov + self.scatter(m, "c_cov")
            pop = self.pop.unsqueeze(0) + self.scatter(m, "c_pop")
            hcol = self.H[:, m["hour"]].t() + self.scatter(m, "c_hour")
            ph = (hcol.clamp_min(0) + 2.0 * self.hprior[m["hour"]].unsqueeze(1)) / (pop.clamp_min(0) + 2.0)
            f["covis"] = torch.log1p(cov.clamp_min(0))
            f["pop"] = torch.log1p(pop.clamp_min(0))
            f["hour"] = torch.log(ph)
        return f

    def _recency(self, m):
        rec = torch.zeros(m["B"], self.P, device=self.device)
        b, p, v = m["sparse"]["rec"]
        if b.numel():
            rec[b, p] = v
        return rec


def sthgcn_filter(ci, qs, train_ids):
    """Keep queries whose target place and user occur in the training split (the test set of
    the supervised baselines on Massive-STEPS)."""
    if not qs:
        return qs
    rows = np.concatenate([ci.trails[i]["rows"] for i in train_ids]) if len(train_ids) else np.zeros(0, int)
    pois = set(np.unique(ci.row_pi[rows]).tolist())
    users = {ci.trails[i]["user_id"] for i in train_ids}
    return [(ti, j) for ti, j in qs
            if int(ci.row_pi[ci.trails[ti]["rows"][j]]) in pois and ci.trails[ti]["user_id"] in users]
