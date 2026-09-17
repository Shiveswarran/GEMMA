import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma.config import Config
from gemma.pretrain import pretrain
def main():
    p=argparse.ArgumentParser()
    for f in ["data_root","fsq_csv","poi_encoder","masking","regularizer","geometry","run_name","device"]:
        p.add_argument("--"+f)
    p.add_argument("--pretrain_cities",nargs="+"); p.add_argument("--epochs",type=int)
    p.add_argument("--batch_size",type=int); p.add_argument("--d_model",type=int)
    p.add_argument("--num_workers",type=int); p.add_argument("--no_amp",action="store_true")
    a=p.parse_args(); cfg=Config()
    for k,v in vars(a).items():
        if v is not None and k!="no_amp": setattr(cfg,k,v)
    if a.no_amp: cfg.amp=False
    pretrain(cfg)
if __name__=="__main__": main()
