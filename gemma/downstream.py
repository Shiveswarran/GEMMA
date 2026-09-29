"""
gemma.downstream
----------------
GEMMA-as-backbone adaptation + evaluation for the Massive-STEPS tasks.

Adaptation modes (shot):
  zero_shot : pretrained backbone only; no target training. Towers + hierarchy
              heads transfer as-is; rank target-city POIs.
  few_shot  : light fine-tune of query/POI towers + heads on a few target trails.
  full      : train on the city's train+validation (in-city supervised setting).

Next-POI ranker (the requested design), score(candidate) =
    w_tower   * z(  q . p_tower  )                       # user-behaviour two-tower
  + w_hier    * z(  logP(act)+logP(pri)+logP(sec)  )     # predicted hierarchy path
  + w_geo     * z( -log1p(dist(last_checkin, cand)) )    # geography (nearest)
  + w_temporal* z(  temporal_fit )                       # time-of-day match
  + w_revisit *   personal_revisit_bonus                 # user behaviour (revisit)
Candidates are RESTRICTED to POIs whose activity is in the predicted top-k
activities (hier_restrict_topk) -> search only the relevant slice of the vocab.

External SOTA integration uses `ExternalAdapter.export(city)` (contract below).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GEMMA
from .data import build_city_data, temporal_features
from .metrics import classification_metrics, ranking_metrics
from .adapter import (AdaptationHead, build_covisitation, user_activity_profile,
                      candidate_features, FEATS)
from .utils import get_logger


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


# =========================================================================== #
#  External SOTA feature-export contract (GETNext/STHGCN/... wrappers use this)
# =========================================================================== #
class ExternalAdapter:
    """`export(city)` returns the tensors a downstream repo injects at its POI
    embedding table (see README 'Integrating a SOTA model'):
        poi_feat (n_pois, d_tower)  : per-POI GEMMA tower embedding
        poi_ids  (n_pois,)          : venue ids aligned to poi_feat
    A repo-specific wrapper maps its own venue-id ordering to poi_ids and
    concatenates poi_feat into its embedding table (3-line patch)."""
    def __init__(self, backbone, cfg, device):
        self.bb, self.cfg, self.device = backbone, cfg, device

    @torch.no_grad()
    def export(self, city):
        pt = city.poi_table
        name = torch.as_tensor(pt["name_emb"]).to(self.device)
        a = torch.as_tensor(pt["act"]).long().to(self.device)
        p = torch.as_tensor(pt["pri"]).long().to(self.device)
        s = torch.as_tensor(pt["sec"]).long().to(self.device)
        feats = []
        for i in range(0, len(a), 4096):
            feats.append(self.bb.poi_embed(name[i:i+4096], a[i:i+4096],
                                           p[i:i+4096], s[i:i+4096]).cpu().numpy())
        return {"poi_feat": np.concatenate(feats), "poi_ids": pt["venue_ids"]}


# =========================================================================== #
def _prefix_batch(city, rows_list, device):
    B = len(rows_list); L = max(len(r) for r in rows_list); cd = city.cont_dim
    cont = np.zeros((B, L, cd), np.float32)
    act = np.zeros((B, L), np.int64); pri = np.zeros((B, L), np.int64)
    sec = np.zeros((B, L), np.int64); mask = np.zeros((B, L), np.float32)
    for i, rows in enumerate(rows_list):
        for j, r in enumerate(rows):
            cont[i, j] = city.cont[r]; act[i, j] = city.act_id[r]
            pri[i, j] = city.pri_id[r]; sec[i, j] = city.sec_id[r]; mask[i, j] = 1
    t = lambda x: torch.as_tensor(x).to(device)
    return {"cont": t(cont), "act": t(act), "pri": t(pri), "sec": t(sec),
            "types": torch.zeros(B, L, dtype=torch.long, device=device),
            "ctx_mask": t(mask)}


def _candidate_bank(bb, city, device):
    pt = city.poi_table
    name = torch.as_tensor(pt["name_emb"]).to(device)
    a = torch.as_tensor(pt["act"]).long().to(device)
    p = torch.as_tensor(pt["pri"]).long().to(device)
    s = torch.as_tensor(pt["sec"]).long().to(device)
    with torch.no_grad():
        P = []
        for i in range(0, len(a), 4096):
            P.append(bb.poi_embed(name[i:i+4096], a[i:i+4096], p[i:i+4096], s[i:i+4096]).cpu().numpy())
    # candidate mean time-of-day (for temporal fit)
    hours = np.zeros(len(pt["venue_ids"]), np.float32)
    df = city.df
    tod = df["timestamp"].dt.hour.values
    for k, v in enumerate(pt["venue_ids"]):
        sel = df["venue_id"].values == v
        hours[k] = tod[sel].mean() if sel.any() else 12.0
    pop = (pt["count"] / pt["count"].sum()).astype(np.float32)
    return {"p": np.concatenate(P), "act": pt["act"], "pri": pt["pri"], "sec": pt["sec"],
            "lat": pt["lat"], "lon": pt["lon"], "hour": hours, "pop": pop,
            "venue_ids": pt["venue_ids"], "id2row": pt["id2row"],
            "codes": _poi_codes(city)}


def _poi_codes(city):
    if city.codes is None:
        return None
    v2c = {}
    vids = city.venue_id
    for i in range(len(vids)):
        v = int(vids[i])
        if v not in v2c:
            v2c[v] = city.codes[i]
    L = city.codes.shape[1]
    out = np.zeros((len(city.poi_table["venue_ids"]), L), np.int64)
    for k, v in enumerate(city.poi_table["venue_ids"]):
        c = v2c.get(int(v))
        if c is not None:
            out[k] = c
    return out


def _few_shot_finetune(bb, city, cfg, device, logger):
    """Light fine-tune of towers + hierarchy heads on a few target trails."""
    import numpy as np
    by_user = {}
    for t in city.trails:
        by_user.setdefault(t["user_id"], []).append(t)
    rng = np.random.RandomState(cfg.seed)
    chosen = []
    for u, ts in by_user.items():
        k = min(cfg.few_shot_trails_per_user, len(ts))
        chosen += [ts[i] for i in rng.choice(len(ts), k, replace=False)]
    chosen = [t for t in chosen if len(t["rows"]) >= 2]
    if not chosen:
        return
    params = (list(bb.query_tower.parameters()) + list(bb.poi_tower.parameters())
              + list(bb.act_head.parameters()) + list(bb.pri_head.parameters())
              + list(bb.sec_head.parameters()) + list(bb.act_emb.parameters())
              + list(bb.pri_emb.parameters()) + list(bb.sec_emb.parameters()))
    opt = torch.optim.AdamW(params, lr=5e-4)
    bb.train()
    for ep in range(cfg.few_shot_epochs):
        rng.shuffle(chosen)
        for s in range(0, len(chosen), 128):
            chunk = chosen[s:s+128]
            b = _prefix_batch(city, [t["rows"][:-1] for t in chunk], device)
            h, summ = bb.extract(b)
            ones = torch.ones(summ.size(0), 1, dtype=torch.long, device=device)
            z1 = bb.tnorm(bb.predictor(summ, ones))[:, 0]
            leaf = [t["rows"][-1] for t in chunk]
            ya = torch.as_tensor([city.act_id[r] for r in leaf]).to(device)
            yp = torch.as_tensor([city.pri_id[r] for r in leaf]).to(device)
            ys = torch.as_tensor([city.sec_id[r] for r in leaf]).to(device)
            la = bb.act_head(z1)
            lp = bb.pri_head(torch.cat([z1, bb.act_emb(ya)], -1))
            ls = bb.sec_head(torch.cat([z1, bb.pri_emb(yp)], -1))
            # tower contrastive on true POI
            name = torch.as_tensor(np.stack([city.name_emb[r] for r in leaf])).to(device)
            q = F.normalize(bb.query_tower(z1), -1)
            ppos = bb.poi_embed(name, ya, yp, ys)
            nce = F.cross_entropy((q @ ppos.t()) / cfg.nce_temp,
                                  torch.arange(len(chunk), device=device))
            loss = F.cross_entropy(la, ya) + F.cross_entropy(lp, yp) + F.cross_entropy(ls, ys) + nce
            opt.zero_grad(); loss.backward(); opt.step()
    bb.eval()
    logger.info(f"[{city.city}] few-shot fine-tuned on {len(chosen)} trails")


def _encode_all(bb, city, trails, device):
    """Encode prefixes -> arrays: act/pri/sec dists, q, z1 (predicted latent)."""
    A=[]; P=[]; S=[]; Z=[]
    for s0 in range(0,len(trails),256):
        chunk=trails[s0:s0+256]
        b=_prefix_batch(city,[t["rows"][:-1] for t in chunk],device)
        with torch.no_grad():
            _,act,pri,sec,q,code,z1=bb.embed_prefix(b)
        A.append(act.cpu().numpy());P.append(pri.cpu().numpy())
        S.append(sec.cpu().numpy());Z.append(z1)
    import numpy as _np
    return (_np.concatenate(A),_np.concatenate(P),_np.concatenate(S),torch.cat(Z,0))


def _prune_keep(act_row, cand_act, topb):
    keep_acts=set(np.argsort(-act_row)[:topb])
    return np.array([a in keep_acts for a in cand_act])


def train_adapter(bb, city, cand, cfg, device, logger, covis):
    """Train the AdaptationHead on the target city's train trails (few/full-shot).
    Backbone frozen; listwise softmax over the pruned candidate set."""
    adapter=AdaptationHead(cfg.adapt_hidden).to(device)
    trails=[t for t in city.trails if len(t["rows"])>=2]
    if not trails:
        return adapter
    opt=torch.optim.AdamW(adapter.parameters(),lr=1e-2,weight_decay=1e-4)
    lat=city.df["latitude"].values.astype(np.float32); lon=city.df["longitude"].values.astype(np.float32)
    A,Pp,Sp,Z=_encode_all(bb,city,trails,device)
    codes_t=torch.as_tensor(cand["codes"]).long().to(device)
    for ep in range(cfg.adapt_epochs):
        order=np.random.RandomState(ep).permutation(len(trails))
        tot=0.0; nb=0
        for bi in range(0,len(order),64):
            batch=order[bi:bi+128]
            loss=torch.zeros((),device=device); used=0
            # AR log-lik for this mini-batch of queries over ALL candidates once
            zb=Z[batch]
            ar=bb.leaf.score_codes(zb, codes_t).cpu().numpy()   # (B, Ncand)
            for bj,ti in enumerate(batch):
                t=trails[ti]; last=t["rows"][-2]
                truth=int(city.venue_id[t["rows"][-1]]); tr=cand["id2row"].get(truth)
                if tr is None: continue
                keep=_prune_keep(A[ti],cand["act"],cfg.adapt_beam_act)
                keep[tr]=True                                   # ensure positive present
                uprof=user_activity_profile(city,t["rows"][:-1],A.shape[1])
                prefix_vids=[int(city.venue_id[r]) for r in t["rows"][:-1]]
                f,idx=candidate_features(cand,ti,A[ti],Pp[ti],Sp[ti],ar[bj],
                    lat[last],lon[last],city.df["timestamp"].iloc[last].hour,
                    prefix_vids,covis,int(city.venue_id[last]),uprof,keep)
                ft=torch.as_tensor(f).to(device)
                scores=adapter(ft)
                pos=int(np.where(idx==tr)[0][0])
                loss=loss+F.cross_entropy(scores.unsqueeze(0),
                     torch.tensor([pos],device=device)); used+=1
            if used>0:
                loss=loss/used
                if torch.isfinite(loss):
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    opt.step()
                    tot+=float(loss.detach()); nb+=1
        logger.info(f"[{city.city}] adapter epoch {ep} loss={tot/max(1,nb):.4f}")
    return adapter


def eval_next_poi(bb, city, cfg, device, logger, adapter=None, covis=None):
    cand=_candidate_bank(bb,city,device)
    codes_t=torch.as_tensor(cand["codes"]).long().to(device)
    eval_trails=[t for t in city.trails if len(t["rows"])>=2]
    lat=city.df["latitude"].values.astype(np.float32); lon=city.df["longitude"].values.astype(np.float32)
    A,Pp,Sp,Z=_encode_all(bb,city,eval_trails,device)
    ranks=[]; a_hit=p_hit=s_hit=0
    fixed=adapter is None
    for s0 in range(0,len(eval_trails),64):
        batch=list(range(s0,min(s0+128,len(eval_trails))))
        zb=Z[batch]
        ar=bb.leaf.score_codes(zb, codes_t).cpu().numpy()       # (B,Ncand) GNPR-SID likelihood
        for bj,ti in enumerate(batch):
            t=eval_trails[ti]; last=t["rows"][-2]
            keep=_prune_keep(A[ti],cand["act"],cfg.adapt_beam_act)
            if keep.sum()<5: keep[:]=True                       # fallback if over-pruned
            uprof=user_activity_profile(city,t["rows"][:-1],A.shape[1])
            prefix_vids=[int(city.venue_id[r]) for r in t["rows"][:-1]]
            f,idx=candidate_features(cand,ti,A[ti],Pp[ti],Sp[ti],ar[bj],
                lat[last],lon[last],city.df["timestamp"].iloc[last].hour,
                prefix_vids,covis,int(city.venue_id[last]),uprof,keep)
            if fixed:
                # equal-weight fusion (zero-shot: no adaptation)
                score=f.sum(1)
            else:
                with torch.no_grad():
                    score=adapter(torch.as_tensor(f).to(device)).cpu().numpy()
            truth=int(city.venue_id[t["rows"][-1]]); tr=cand["id2row"].get(truth)
            if tr is None or tr not in idx:
                rank=len(cand["venue_ids"])
            else:
                order_local=np.argsort(-score)
                pos_in_keep=int(np.where(idx==tr)[0][0])
                rank=int(np.where(order_local==pos_in_keep)[0][0])
            ranks.append(rank)
            tleaf=t["rows"][-1]
            a_hit+=int(A[ti].argmax()==city.act_id[tleaf])
            p_hit+=int(Pp[ti].argmax()==city.pri_id[tleaf])
            s_hit+=int(Sp[ti].argmax()==city.sec_id[tleaf])
    met=ranking_metrics(ranks); N=max(1,len(ranks))
    met.update({"n_trails":len(ranks),"n_candidates":len(cand["venue_ids"]),
                "act@1":a_hit/N,"pri@1":p_hit/N,"sec@1":s_hit/N})
    return met


def eval_next_activity(bb, city, cfg, device):
    eval_trails = [t for t in city.trails if len(t["rows"]) >= 2]
    yt, yp = [], []
    for s0 in range(0, len(eval_trails), 256):
        chunk = eval_trails[s0:s0+256]
        b = _prefix_batch(city, [t["rows"][:-1] for t in chunk], device)
        with torch.no_grad():
            _, act_d, _, _, _, _, _ = bb.embed_prefix(b)
        yp += list(act_d.argmax(-1).cpu().numpy())
        yt += [city.act_id[t["rows"][-1]] for t in chunk]
    return classification_metrics(yt, yp)


def run_downstream(bb, fsq, enc, cfg, city_name, run_dir, task="next_poi",
                   shot="zero_shot", logger=None):
    logger = logger or get_logger(f"gemma.down.{city_name}")
    device = cfg.resolved_device()

    test = build_city_data(cfg, city_name, "test", fsq, enc, logger, filter_users=False)
    if task == "next_activity":
        met = eval_next_activity(bb, test, cfg, device)
        met.update({"city": city_name, "task": task, "shot": shot})
        logger.info(f"[{city_name}/{task}/{shot}] " +
                    " ".join(f"{k}={round(v,4)}" for k, v in met.items() if isinstance(v, float)))
        with open(os.path.join(run_dir, f"down_{city_name}_{task}_{shot}.json"), "w") as f:
            json.dump(met, f, indent=2)
        return met

    # ---- next-POI: hierarchical decoding + Adaptation Head ----
    adapter, covis = None, None
    if shot in ("few_shot", "full"):
        splits = ["train"] if shot == "few_shot" else ["train", "validation"]
        train = build_city_data(cfg, city_name, splits, fsq, enc, logger, filter_users=False)
        covis = build_covisitation(train)                    # collaborative signal
        cand_tr = _candidate_bank(bb, train, device)
        adapter = train_adapter(bb, train, cand_tr, cfg, device, logger, covis)
        # co-visitation is estimated from target-train; carry it to the test candidates
        covis_test = covis
    else:
        covis_test = None                                    # zero-shot: no target stats

    met = eval_next_poi(bb, test, cfg, device, logger, adapter=adapter, covis=covis_test)
    met.update({"city": city_name, "task": task, "shot": shot})
    logger.info(f"[{city_name}/{task}/{shot}] " +
                " ".join(f"{k}={round(v,4)}" for k, v in met.items() if isinstance(v, float)))
    with open(os.path.join(run_dir, f"down_{city_name}_{task}_{shot}.json"), "w") as f:
        json.dump(met, f, indent=2)
    return met
