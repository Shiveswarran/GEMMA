#!/usr/bin/env bash
# GEMMA-HREM experiments on the single GPU 0, one after another.
#   nohup bash scripts/run_hrem.sh > run_hrem.out 2>&1 &   ;   tail -f run_hrem.out
# Set ALL_CITIES=1 to also run all 15 cities at the end (long).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/scratch/umni5/a/$USER/hf}; export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
CSV=fsk_categories.csv
SRC="istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"
ORDER="tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung kuwait_city sao_paulo new_york jakarta petaling_jaya istanbul"
mkdir -p logs

python - <<'PY' || { echo "No usable GPU -> stopping."; exit 1; }
import torch, sys
ok = torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0) if ok else "NONE")
sys.exit(0 if ok else 1)
PY

echo "=== 0. revisit rate by activity (CPU, motivates the activity-conditioned gate) ($(date)) ==="
python scripts/diagnose_caps.py --data_root data --cities $ORDER --fsq_csv $CSV --skip_poi \
  --out logs/diag_act2.json 2>&1 | tee logs/diag_act2.log

echo "=== 1. HREM source pretraining on the 6 source cities (v4 backbone) ($(date)) ==="
python scripts/train_realizer_source.py --run_dir runs/gemma_v4 --data_root data --cities $SRC \
  --tune_backbone --retriever hrem 2>&1 | tee logs/zs_hrem_v4.log

echo "=== 2. Tokyo: zero / few (5%) / full, both protocols ($(date)) ==="
python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data --cities tokyo \
  --retriever hrem --shots zero_shot few_shot full --few_shot_frac 0.05 2>&1 | tee logs/eval_hrem_tokyo.log

echo "=== 3. ablation: HREM without JEPA pretraining ($(date)) ==="
python scripts/train_realizer_source.py --run_dir runs/abl_random --data_root data --cities $SRC \
  --tune_backbone --random_backbone --retriever hrem 2>&1 | tee logs/zs_hrem_random.log
python scripts/run_downstream.py --run_dir runs/abl_random --data_root data --cities tokyo \
  --retriever hrem --shots zero_shot full --random_backbone 2>&1 | tee logs/eval_hrem_random_tokyo.log

echo "=== 4. ablation: in-city only (no source transfer, no pretraining) ($(date)) ==="
mkdir -p runs/abl_incity && cp runs/gemma_v4/config.json runs/gemma_v4/gemma.pt runs/abl_incity/
python scripts/run_downstream.py --run_dir runs/abl_incity --data_root data --cities tokyo \
  --retriever hrem --shots full --random_backbone --no_init_from_zs 2>&1 | tee logs/eval_hrem_incity_tokyo.log

if [ "${ALL_CITIES:-0}" = "1" ]; then
  echo "=== 5. all 15 cities ($(date)) ==="
  python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data --cities $ORDER \
    --retriever hrem --shots zero_shot few_shot full --few_shot_frac 0.05 2>&1 | tee logs/eval_hrem_all.log
fi
echo "=== DONE ($(date)) ==="
grep -h "FULL-VOCAB" logs/eval_hrem_*tokyo.log | cut -c1-300
