import argparse, sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.config import Config
from gemma.fsq import FSQTaxonomy
from gemma.poi_encoder import POIEncoder
from gemma.backbone import GEMMA
from gemma.downstream import run_downstream
from gemma.utils import get_logger
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run_dir",required=True); p.add_argument("--data_root")
    p.add_argument("--cities",nargs="+",required=True)
    p.add_argument("--tasks",nargs="+",default=["next_poi","next_activity"])
    p.add_argument("--shots",nargs="+",default=["zero_shot","few_shot","full"],
                   choices=["zero_shot","few_shot","full"])
    p.add_argument("--device")
    a=p.parse_args()
    cfg=Config.from_json(os.path.join(a.run_dir,"config.json"))
    if a.data_root: cfg.data_root=a.data_root
    if a.device: cfg.device=a.device
    logger=get_logger("gemma.down", os.path.join(a.run_dir,"downstream.log"))
    fsq=FSQTaxonomy(cfg.fsq_csv); enc=POIEncoder(cfg,logger); cfg.poi_emb_dim=enc.dim
    ck=torch.load(os.path.join(a.run_dir,"gemma.pt"),map_location="cpu",weights_only=False)
    rows=[]
    for city in a.cities:
        for task in a.tasks:
            for shot in a.shots:
                # reload a FRESH backbone per (few_shot/full) so target fine-tune doesn't leak
                bb=GEMMA(cfg,ck["cont_dim"],ck["n_act"],ck["n_pri"],ck["n_sec"]).to(cfg.resolved_device())
                bb.load_state_dict(ck["model"]); bb.eval()
                m=run_downstream(bb,fsq,enc,cfg,city,a.run_dir,task=task,shot=shot,logger=logger)
                rows.append(m)
    with open(os.path.join(a.run_dir,"benchmark_summary.json"),"w") as f:
        json.dump(rows,f,indent=2)
    logger.info("=== BENCHMARK SUMMARY ===")
    for m in rows:
        if m["task"]=="next_poi":
            logger.info(f"{m['city']:>14} next_poi/{m['shot']:<9} acc@1={m['acc@1']:.4f} "
                        f"acc@5={m['acc@5']:.4f} ndcg@5={m['ndcg@5']:.4f} mrr={m['mrr']:.4f} "
                        f"| act@1={m['act@1']:.3f} pri@1={m['pri@1']:.3f} sec@1={m['sec@1']:.3f}")
        else:
            logger.info(f"{m['city']:>14} next_activity/{m['shot']:<9} "
                        f"acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f}")
if __name__=="__main__": main()
