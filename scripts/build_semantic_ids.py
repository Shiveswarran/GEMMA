"""Offline: build POI feature matrix, train an RQ-VAE on the pretrain cities'
POIs (shared -> transferable codebooks), then quantize EVERY listed city's POIs
into an L-level semantic ID.  Caches {city}_codes.npz {venue_ids, codes} + the
RQ-VAE state.  Run ONCE before pretraining with use_semantic_id=True."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.gpu_env import pin_gpu; pin_gpu()      # before torch is imported
import numpy as np, torch
from gemma.config import Config
from gemma.fsq import FSQTaxonomy
from gemma.poi_encoder import POIEncoder
from gemma.data import build_city_data
from gemma.semantic_id import RQVAE
from gemma.utils import get_logger, set_seed

def poi_features(city):
    """Block-balanced feature vector.  Missing-coordinate POIs (Massive-STEPS
    privacy fallback) get a deterministic venue-id hash in the geo block + a
    coord-valid flag, so they stay DISTINCT instead of all collapsing to (0,0)."""
    pt = city.poi_table
    def l2(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    name = l2(pt["name_emb"].astype(np.float32))
    lat = pt["lat"].astype(np.float32); lon = pt["lon"].astype(np.float32)
    valid = pt.get("coord_valid", np.ones(len(lat), np.float32))
    clat, clon = np.nanmean(lat[valid > 0.5]) if (valid > 0.5).any() else 0.0, \
                 np.nanmean(lon[valid > 0.5]) if (valid > 0.5).any() else 0.0
    geo_feats = []
    for scale in (1.0, 8.0, 64.0):
        geo_feats += [np.sin((lat - clat) * scale), np.cos((lat - clat) * scale),
                      np.sin((lon - clon) * scale), np.cos((lon - clon) * scale)]
    geo = np.stack(geo_feats, 1).astype(np.float32)
    # per-POI hash fallback where coords are missing -> distinct, not collapsed
    vids = pt["venue_ids"].astype(np.int64)
    rng = np.random.RandomState(0)
    hashv = np.stack([np.sin(vids * (k + 1) * 0.9161 + k) for k in range(geo.shape[1])], 1).astype(np.float32)
    geo = np.where(valid[:, None] > 0.5, geo, hashv)
    geo = l2(geo)
    h = pt.get("hour"); h = np.full(len(lat), 12.0, np.float32) if h is None else h
    tod = l2(np.stack([np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)], 1).astype(np.float32))
    hier = l2(np.stack([pt["act"], pt["pri"], pt["sec"]], 1).astype(np.float32) + 1.0)
    flag = valid[:, None].astype(np.float32)                          # coord-valid flag
    return np.concatenate([1.0*name, 1.0*geo, 0.5*tod, 0.5*hier, flag], 1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",default="data"); ap.add_argument("--fsq_csv",required=True)
    ap.add_argument("--train_cities",nargs="+",required=True)   # pretrain cities -> shared codebooks
    ap.add_argument("--all_cities",nargs="+",required=True)     # quantize these
    ap.add_argument("--poi_encoder",default="llm_cached"); ap.add_argument("--poi_emb_cache",default="poi_emb_cache")
    ap.add_argument("--levels",type=int,default=3); ap.add_argument("--codebook",type=int,default=256)
    ap.add_argument("--epochs",type=int,default=40); ap.add_argument("--cache",default="sid_cache")
    ap.add_argument("--bottleneck",type=int,default=32)
    ap.add_argument("--gamma",type=float,default=2.0); ap.add_argument("--ema",type=float,default=0.95)
    ap.add_argument("--device",default="auto")
    a=ap.parse_args(); set_seed(42); os.makedirs(a.cache,exist_ok=True)
    cfg=Config(data_root=a.data_root,fsq_csv=a.fsq_csv,poi_encoder=a.poi_encoder,
               poi_emb_cache=a.poi_emb_cache,use_semantic_id=False)
    logger=get_logger("gemma.sid")
    fsq=FSQTaxonomy(a.fsq_csv); enc=POIEncoder(cfg,logger); cfg.poi_emb_dim=enc.dim
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device

    # gather training POI features from the pretrain cities
    feats=[]
    citydata={}
    for c in set(a.train_cities)|set(a.all_cities):
        cd=build_city_data(cfg,c,["train"],fsq,enc,logger,filter_users=False)
        citydata[c]=cd
    Xtr=np.concatenate([poi_features(citydata[c]) for c in a.train_cities])
    logger.info(f"training RQ-VAE on {len(Xtr)} POIs, feat_dim={Xtr.shape[1]}")
    model=RQVAE(Xtr.shape[1], d=a.bottleneck, n_levels=a.levels, codebook_size=a.codebook, gamma=a.gamma, ema=a.ema).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    Xt=torch.as_tensor(Xtr).to(dev)
    model.kmeans_init(Xt[torch.randperm(len(Xt))[:4096]])
    import copy
    best_score, best_state, best_ep = -1.0, None, -1
    Xeval = Xt[torch.randperm(len(Xt))[:20000]]              # fixed-ish eval subset
    for ep in range(a.epochs):
        model.train()
        perm=torch.randperm(len(Xt))
        tot=0
        for s in range(0,len(Xt),4096):
            idx=perm[s:s+4096]; out=model(Xt[idx])
            opt.zero_grad(); out["loss"].backward(); opt.step(); tot+=float(out["loss"].detach())
        model.revive_dead(Xt[torch.randperm(len(Xt))[:8192]])
        # selection metric = # DISTINCT semantic IDs on the eval subset (not loss!)
        model.eval()
        with torch.no_grad():
            codes_e = model.encode_codes(Xeval).cpu().numpy()
        distinct = len({tuple(c) for c in codes_e})
        score = distinct / len(codes_e)                     # fraction unique (higher=better)
        if score > best_score:
            best_score, best_ep = score, ep
            best_state = copy.deepcopy(model.state_dict())
        if ep%2==0 or ep==a.epochs-1:
            logger.info(f"ep{ep} loss={tot:.1f} recon={float(out['recon']):.4f} "
                        f"kl={float(out['kl']):.2f} usage(lvl0)={out['usage']:.2f} "
                        f"distinct={score:.3f} (best ep{best_ep}={best_score:.3f})")
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"restoring best checkpoint from ep{best_ep} "
                    f"(distinct-fraction={best_score:.3f})")
    model.eval()
    torch.save(model.state_dict(), os.path.join(a.cache,"rqvae.pt"))
    # quantize every city with the FROZEN codebooks
    model.eval()
    for c in a.all_cities:
        X=torch.as_tensor(poi_features(citydata[c])).to(dev)
        with torch.no_grad():
            codes=model.encode_codes(X).cpu().numpy().astype(np.int64)
        np.savez(os.path.join(a.cache,f"{c}_codes.npz"),
                 venue_ids=citydata[c].poi_table["venue_ids"], codes=codes)
        uniq=len(set(map(tuple,codes)))
        logger.info(f"[{c}] {len(codes)} POIs -> {uniq} distinct semantic IDs "
                    f"(collision rate {1-uniq/len(codes):.2%})")
    logger.info(f"done -> {a.cache}")
if __name__=="__main__": main()
