"""Feasibility test for a TRAINING-FREE CITY ADAPTER: does reading the target city's
check-in log as an episodic memory (no gradient step in the target city) close the gap
to STHGCN, and does the pretrained backbone make that memory better?

For each held-out city:
  * zero-shot GEMMA (HREM) scores  log P_hrem(v | q)                       [as before]
  * memory M = {(key(c_i), v_i)}: every next-step transition in the TRAIN log of the city
    (all users, all positions; the same data STHGCN trains on), key = backbone context
    latent [z1, h_last, summary] computed by the frozen source-trained model.
  * three memory read-outs
      markov : P(v | last POI) from city transition counts (+ popularity back-off)   (no backbone)
      proto  : P(v) ~ exp(tau * cos(key_q, mu_v)),  mu_v = mean key of contexts followed by v
      knn    : P(v) ~ sum_{i in topK(key_q)} exp(tau * cos(key_q, key_i)) [v_i = v]  (kNN-LM)
  * product of experts  score = log P_hrem + beta * log(P_mem + eps);  beta, tau picked on
    the VALIDATION split (only hyper-parameters; nothing is trained).
Reports test Acc@1 / NDCG@5 per city for every read-out next to STHGCN, and saves ranks
for paired tests (ranks_mem_<variant>_<city>.npy, 0-based).

python scripts/city_memory_eval.py --run_dir runs/gemma_v4 --data_root data \
    --cities tokyo shanghai melbourne sydney moscow
python scripts/city_memory_eval.py --run_dir runs/abl_random --random_backbone ...   # backbone ablation
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.gpu_env import pin_gpu; pin_gpu()
import numpy as np
import torch
import torch.nn.functional as F
from gemma.downstream import load_config, load_backbone, split_ids, ALL_SPLITS
from gemma.data import build_city_data
from gemma.fsq import FSQTaxonomy
from gemma.poi_encoder import POIEncoder
from gemma.realize import CityIndex, new_head, _forward, _amp, sthgcn_filter
from gemma.metrics import ranking_metrics
from gemma.sota_ref import SUP
from gemma.utils import check_device, get_logger, set_seed

EPS = 1e-6


@torch.no_grad()
def run_queries(bb, head, ci, qs, cfg, bs, want_logits=True):
    keys, logits, tgt, last = [], [], [], []
    for s in range(0, len(qs), bs):
        out = _forward(bb, head, ci, qs[s:s + bs], False, _amp(cfg, ci.device), 0)
        keys.append(F.normalize(out["key"].float(), dim=-1).half())
        if want_logits:
            logits.append(out["logits"].float().cpu())
        tgt.append(out["m"]["tgt_pi"]); last.append(out["m"]["last_pi"])
    return (torch.cat(keys), torch.cat(logits) if want_logits else None,
            torch.cat(tgt), torch.cat(last))


def mem_markov(last_q, mk_last, mk_tgt, P, alpha=1.0):
    """city first-order transitions with popularity back-off: (B,P) probabilities."""
    dv = last_q.device
    T = torch.zeros(P, P, device=dv) if P <= 12000 else None
    pop = torch.bincount(mk_tgt, minlength=P).float() + 1.0
    pop = pop / pop.sum()
    if T is not None:
        T.index_put_((mk_last, mk_tgt), torch.ones_like(mk_tgt, dtype=torch.float), accumulate=True)
        rows = T[last_q]
    else:                                    # big cities: sparse rows on demand
        rows = torch.zeros(len(last_q), P, device=dv)
        order = torch.argsort(mk_last)
        sl, st = mk_last[order], mk_tgt[order]
        for b, l in enumerate(last_q.tolist()):
            lo = torch.searchsorted(sl, torch.tensor(l, device=dv))
            hi = torch.searchsorted(sl, torch.tensor(l, device=dv), right=True)
            if hi > lo:
                rows[b].index_add_(0, st[lo:hi], torch.ones(int(hi - lo), device=dv))
    n = rows.sum(1, keepdim=True)
    return (rows + alpha * pop) / (n + alpha)


def mem_proto(key_q, mk_key, mk_tgt, P, tau):
    D = mk_key.size(1)
    mu = torch.zeros(P, D, device=key_q.device)
    mu.index_add_(0, mk_tgt, mk_key.float())
    has = torch.bincount(mk_tgt, minlength=P) > 0
    mu = F.normalize(mu, dim=-1)
    s = tau * key_q.float() @ mu.t()
    s = s.masked_fill(~has, float("-inf"))
    return torch.softmax(s, -1)


def mem_knn(key_q, mk_key, mk_tgt, P, tau, K=64, chunk=256):
    outs = []
    for s0 in range(0, key_q.size(0), chunk):
        sim = key_q[s0:s0 + chunk].float() @ mk_key.float().t()
        v, i = sim.topk(min(K, sim.size(1)), dim=1)
        w = torch.softmax(tau * v, -1)
        p = torch.zeros(sim.size(0), P, device=key_q.device)
        p.scatter_add_(1, mk_tgt[i], w)
        outs.append(p)
    return torch.cat(outs)


def ranks_of(score, tgt):
    st = score.gather(1, tgt.unsqueeze(1))
    return (score > st).sum(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--data_root")
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--random_backbone", action="store_true")
    ap.add_argument("--mem_max", type=int, default=150000)
    ap.add_argument("--val_max", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--hist_last", type=int, default=-1)
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config(a.run_dir)
    if a.data_root: cfg.data_root = a.data_root
    cfg.random_backbone = a.random_backbone; cfg.allow_cpu = a.allow_cpu
    cfg.retriever = "hrem"; cfg.hist_last = a.hist_last
    logger = get_logger("gemma.mem", os.path.join(a.run_dir, f"city_memory{a.tag}.log"))
    device = check_device(cfg, logger); set_seed(cfg.seed)
    fsq = FSQTaxonomy(cfg.fsq_csv); enc = POIEncoder(cfg, logger); cfg.poi_emb_dim = enc.dim
    zs = torch.load(os.path.join(a.run_dir, "zs_hrem.pt"), map_location="cpu", weights_only=False)
    rows = []
    for c in a.cities:
        t0 = time.time()
        city = build_city_data(cfg, c, ALL_SPLITS, fsq, enc, logger, filter_users=False)
        ids = split_ids(city)
        bb = load_backbone(cfg, a.run_dir, device, zs.get("backbone"))
        ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device=device)
        head = new_head(bb, ci, cfg, device); head.load_state_dict(zs["head"], strict=False)
        head.eval()
        rng = np.random.RandomState(cfg.seed)
        q_mem = ci.queries(ids["train"], True)
        if len(q_mem) > a.mem_max:
            q_mem = [q_mem[i] for i in rng.permutation(len(q_mem))[:a.mem_max]]
        q_val = ci.queries(ids["validation"], False)
        q_val = [q_val[i] for i in rng.permutation(len(q_val))[:a.val_max]]
        q_te = ci.queries(ids["test"], False)
        if getattr(cfg, "eval_protocol", "sthgcn") == "sthgcn":       # same test set as STHGCN
            q_val = sthgcn_filter(ci, q_val, ids["train"]); q_te = sthgcn_filter(ci, q_te, ids["train"])
        mk_key, _, mk_tgt, mk_last = run_queries(bb, head, ci, q_mem, cfg, a.bs, want_logits=False)
        kv, lv, tv, lastv = run_queries(bb, head, ci, q_val, cfg, a.bs)
        kt, lt, tt, lastt = run_queries(bb, head, ci, q_te, cfg, a.bs)
        P = ci.P
        logger.info(f"[{c}] memory={len(q_mem)} val={len(q_val)} test={len(q_te)} POIs={P} "
                    f"({time.time() - t0:.0f}s)")

        def read(kind, key, last, tau):
            if kind == "markov":
                return mem_markov(last, mk_last, mk_tgt, P)
            if kind == "proto":
                return mem_proto(key, mk_key, mk_tgt, P, tau)
            return mem_knn(key, mk_key, mk_tgt, P, tau)

        res = {"city": c}
        base_rk = ranks_of(lt.to(device), tt)
        res["hrem"] = ranking_metrics(base_rk)
        np.save(os.path.join(a.run_dir, f"ranks_mem_hrem_{c}{a.tag}.npy"), base_rk)
        for kind in ("markov", "proto", "knn"):
            taus = [1.0] if kind == "markov" else [10.0, 20.0, 50.0]
            best = (-1, None)
            for tau in taus:
                pv = read(kind, kv, lastv, tau)
                for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
                    sc = lv.to(device) + beta * torch.log(pv + EPS)
                    acc = float((ranks_of(sc, tv) < 1).mean())
                    if acc > best[0]:
                        best = (acc, (tau, beta))
            tau, beta = best[1]
            pt = read(kind, kt, lastt, tau)
            rk = ranks_of(lt.to(device) + beta * torch.log(pt + EPS), tt)
            alone = ranks_of(torch.log(pt + EPS), tt)
            res[kind] = dict(ranking_metrics(rk), tau=tau, beta=beta, val_acc1=best[0])
            res[kind + "_alone"] = ranking_metrics(alone)
            np.save(os.path.join(a.run_dir, f"ranks_mem_{kind}_{c}{a.tag}.npy"), rk)
        sup = SUP.get(c, (float("nan"),) * 2)
        logger.info(f"[{c}] Acc@1/NDCG@5  hrem={res['hrem']['acc@1']:.4f}/{res['hrem']['ndcg@5']:.4f} | "
                    + " | ".join(f"+{k}={res[k]['acc@1']:.4f}/{res[k]['ndcg@5']:.4f}"
                                 f"(b={res[k]['beta']},t={res[k]['tau']}; alone {res[k + '_alone']['acc@1']:.4f})"
                                 for k in ("markov", "proto", "knn"))
                    + f" || STHGCN {sup[0]:.3f}/{sup[1]:.3f}")
        rows.append(res)
        del bb, head, ci, mk_key
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    with open(os.path.join(a.run_dir, f"city_memory{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("=== SUMMARY Acc@1 (test)  city | hrem | +markov | +proto | +knn | STHGCN")
    for r in rows:
        logger.info(f"{r['city']:>12} | {r['hrem']['acc@1']:.3f} | {r['markov']['acc@1']:.3f} | "
                    f"{r['proto']['acc@1']:.3f} | {r['knn']['acc@1']:.3f} | {SUP.get(r['city'], (float('nan'),))[0]:.3f}")


if __name__ == "__main__":
    main()
