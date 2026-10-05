import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.gpu_env import pin_gpu; pin_gpu()      # before torch is imported
from gemma.config import Config
from gemma.pretrain import pretrain
def main():
    p=argparse.ArgumentParser()
    for f in ["data_root","fsq_csv","poi_encoder","masking","regularizer","geometry","run_name","device","leaf_head","sid_cache","predictor_type","out_dir"]:
        p.add_argument("--"+f)
    p.add_argument("--pretrain_cities",nargs="+"); p.add_argument("--epochs",type=int)
    p.add_argument("--batch_size",type=int); p.add_argument("--d_model",type=int)
    p.add_argument("--num_workers",type=int); p.add_argument("--no_amp",action="store_true")
    p.add_argument("--allow_cpu",action="store_true")
    p.add_argument("--sid_levels",type=int); p.add_argument("--sid_codebook",type=int)
    p.add_argument("--no_semantic_id",action="store_true"); p.add_argument("--no_uncertainty",action="store_true")
    a=p.parse_args(); cfg=Config()
    for k,v in vars(a).items():
        if v is not None and k not in ("no_amp","allow_cpu"): setattr(cfg,k,v)
    if a.no_amp: cfg.amp=False
    cfg.allow_cpu=a.allow_cpu
    if a.no_semantic_id: cfg.use_semantic_id=False
    if a.no_uncertainty: cfg.uncertainty_weighting=False
    pretrain(cfg)
if __name__=="__main__": main()
