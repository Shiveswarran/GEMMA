#!/usr/bin/env bash
# Runs experiments A -> B -> C (-> D) one after another on the single GPU 0.
# usage (from the repo root):  nohup bash scripts/run_abc.sh > run_abc.out 2>&1 &
#                              tail -f run_abc.out
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/scratch/umni5/a/$USER/hf}; export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
CSV=fsk_categories.csv
SRC="istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"
ALL="tokyo istanbul jakarta new_york kuwait_city petaling_jaya sao_paulo bandung beijing melbourne moscow palembang shanghai sydney tangerang"
mkdir -p logs

python - <<'PY' || { echo "No usable GPU -> stopping."; exit 1; }
import torch, sys
ok = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if ok else "NONE", "| torch", torch.__version__)
sys.exit(0 if ok else 1)
PY

# keep the v5b results (A overwrites them)
[ -f runs/gemma_v4/zs_realizer.pt ] && [ ! -f runs/gemma_v4/zs_realizer_v5b.pt ] && \
  cp runs/gemma_v4/zs_realizer.pt runs/gemma_v4/zs_realizer_v5b.pt
mkdir -p runs/gemma_v4/v5b && cp -n runs/gemma_v4/v5_*.json runs/gemma_v4/v5b/ 2>/dev/null || true

echo "=== A: v4 backbone, fixed realizer ($(date)) ==="
python scripts/train_realizer_source.py --run_dir runs/gemma_v4 --data_root data \
  --cities $SRC --tune_backbone 2>&1 | tee logs/zs_v4c.log
python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data --cities tokyo \
  --shots zero_shot few_shot full --few_shot_frac 0.05 2>&1 | tee logs/eval_v5c_tokyo.log

echo "=== B: v5 backbone ($(date)) ==="
python scripts/train_realizer_source.py --run_dir runs/gemma_v5 --data_root data \
  --cities $SRC --tune_backbone 2>&1 | tee logs/zs_v5.log
python scripts/run_downstream.py --run_dir runs/gemma_v5 --data_root data --cities tokyo \
  --shots zero_shot few_shot full --few_shot_frac 0.05 2>&1 | tee logs/eval_v5_backbone_tokyo.log

echo "=== C: no-pretraining ablation ($(date)) ==="
mkdir -p runs/abl_random && cp runs/gemma_v4/config.json runs/gemma_v4/gemma.pt runs/abl_random/
rm -f runs/abl_random/zs_realizer.pt
python scripts/train_realizer_source.py --run_dir runs/abl_random --data_root data \
  --cities $SRC --tune_backbone --random_backbone 2>&1 | tee logs/zs_random.log
python scripts/run_downstream.py --run_dir runs/abl_random --data_root data --cities tokyo \
  --shots zero_shot full --random_backbone 2>&1 | tee logs/eval_abl_random_tokyo.log

echo "=== D: next-activity baselines (CPU) ($(date)) ==="
python scripts/diagnose_caps.py --data_root data --cities $ALL --fsq_csv $CSV --skip_poi \
  --out logs/diag_act.json 2>&1 | tee logs/diag_act.log

echo "=== ALL DONE ($(date)) ==="
grep -h "FULL-VOCAB" logs/eval_v5c_tokyo.log logs/eval_v5_backbone_tokyo.log logs/eval_abl_random_tokyo.log | cut -c1-260
