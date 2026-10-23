#!/usr/bin/env bash
# GEMMA-WM (HS-JEPA world model) experiments on the single GPU 0, one after another.
#   nohup bash scripts/run_wm.sh > run_wm.out 2>&1 &   ;   tail -f run_wm.out
#   ABLATIONS=1  -> also the objective ablations (each is a full pretraining run)
#   ALL_CITIES=1 -> also the 15-city evaluation of the main model
#   SMOKE=1      -> 1-epoch / 2k-queries-per-city crash test (~10 min), then exit
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/scratch/umni5/a/$USER/hf}; export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
SRC="istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"
ORDER="tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung kuwait_city sao_paulo new_york jakarta petaling_jaya istanbul"
BUDGET="--epochs ${EPOCHS:-6} --max_queries_per_city ${QPC:-150000} --batch 128"
# v6 lesson: the head plateaus inside epoch 0 and patience=3 @ 1000 noisy val queries/city
# stopped training in epoch 1 -- far too early for a from-scratch world model.
SCHED="--min_epochs ${MIN_EPOCHS:-3} --patience 4 --val_every 2000 --val_per_city 2000"
mkdir -p logs

python - <<'PY' || { echo "No usable GPU -> stopping."; exit 1; }
import torch, sys
ok = torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0) if ok else "NONE")
sys.exit(0 if ok else 1)
PY

newdir () {  # architecture/config template only (weights are re-initialised with --random_backbone)
  mkdir -p "$1" && cp runs/gemma_v4/config.json runs/gemma_v4/gemma.pt "$1"/
}

if [ "${SMOKE:-0}" = "1" ]; then
  newdir runs/wm_smoke
  python scripts/train_realizer_source.py --run_dir runs/wm_smoke --data_root data --cities $SRC \
    --retriever wm --tune_backbone --random_backbone --lr_backbone 3e-4 --lr_head 1e-3 \
    --epochs 1 --max_queries_per_city 2000 --batch 128 --val_per_city 200 --log_every 20 2>&1 | tee logs/smoke_wm.log
  python scripts/run_downstream.py --run_dir runs/wm_smoke --data_root data --cities tokyo \
    --retriever wm --random_backbone --shots zero_shot --protocols full 2>&1 | tee -a logs/smoke_wm.log
  exit 0
fi

echo "=== 0. re-evaluate the v6 HREM checkpoint with the new diagnostics + per-query ranks ($(date)) ==="
python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data --cities tokyo \
  --retriever hrem --shots zero_shot full 2>&1 | tee logs/eval_hrem_tokyo_diag.log

echo "=== 1. GEMMA-WM pretraining from scratch on the 6 source cities ($(date)) ==="
newdir runs/gemma_wm
python scripts/train_realizer_source.py --run_dir runs/gemma_wm --data_root data --cities $SRC \
  --retriever wm --tune_backbone --random_backbone --lr_backbone 3e-4 --lr_head 1e-3 $BUDGET $SCHED \
  2>&1 | tee logs/pretrain_wm.log

echo "=== 2. Tokyo: zero / few (5%) / full ($(date)) ==="
python scripts/run_downstream.py --run_dir runs/gemma_wm --data_root data --cities tokyo \
  --retriever wm --random_backbone --shots zero_shot few_shot full --few_shot_frac 0.05 \
  2>&1 | tee logs/eval_wm_tokyo.log

echo "=== 2b. SAME world model trained in Tokyo only (no transfer) -- the decisive baseline ($(date)) ==="
newdir runs/gemma_wm_incity
python scripts/run_downstream.py --run_dir runs/gemma_wm_incity --data_root data --cities tokyo \
  --retriever wm --random_backbone --shots full --ft_epochs 30 --ft_lr_backbone 3e-4 \
  --patience 5 --min_epochs 10 2>&1 | tee logs/eval_wm_incity_tokyo.log

echo "=== 3. GEMMA-WM initialised from the old JEPA backbone (does v4 JEPA add anything?) ($(date)) ==="
newdir runs/gemma_wm_v4init
python scripts/train_realizer_source.py --run_dir runs/gemma_wm_v4init --data_root data --cities $SRC \
  --retriever wm --tune_backbone --lr_backbone 1e-4 --lr_head 1e-3 $BUDGET $SCHED 2>&1 | tee logs/pretrain_wm_v4init.log
python scripts/run_downstream.py --run_dir runs/gemma_wm_v4init --data_root data --cities tokyo \
  --retriever wm --shots zero_shot full 2>&1 | tee logs/eval_wm_v4init_tokyo.log

if [ "${ABLATIONS:-0}" = "1" ]; then
  for AB in "noJEPA:--lambda_wm_jepa 0 --lambda_wm_infill 0" "noRetAux:--lambda_wm_lat 0 --lambda_wm_ptr 0" \
            "flat:--wm_flat" "H1:--wm_H 1" "noInfill:--wm_infill 0"; do
    NAME=${AB%%:*}; FLAGS=${AB#*:}
    echo "=== ablation $NAME ($(date)) ==="
    newdir runs/abl_wm_$NAME
    python scripts/train_realizer_source.py --run_dir runs/abl_wm_$NAME --data_root data --cities $SRC \
      --retriever wm --tune_backbone --random_backbone --lr_backbone 3e-4 --lr_head 1e-3 $BUDGET $SCHED $FLAGS \
      2>&1 | tee logs/pretrain_abl_$NAME.log
    python scripts/run_downstream.py --run_dir runs/abl_wm_$NAME --data_root data --cities tokyo \
      --retriever wm --random_backbone --shots zero_shot full 2>&1 | tee logs/eval_abl_${NAME}_tokyo.log
  done
fi

if [ "${ALL_CITIES:-0}" = "1" ]; then
  echo "=== 15 cities ($(date)) ==="
  python scripts/run_downstream.py --run_dir runs/gemma_wm --data_root data --cities $ORDER \
    --retriever wm --random_backbone --shots zero_shot few_shot full --few_shot_frac 0.05 \
    2>&1 | tee logs/eval_wm_all.log
fi
echo "=== DONE ($(date)) ==="
grep -h "FULL-VOCAB" logs/eval_hrem_tokyo_diag.log logs/eval_wm*_tokyo.log logs/eval_abl_*_tokyo.log 2>/dev/null | cut -c1-420
echo "=== paired tests (Tokyo, same queries) ==="
pt () { [ -f "$1" ] && [ -f "$2" ] && { echo "--- $3"; python scripts/paired_test.py "$1" "$2"; }; }
pt runs/gemma_wm/ranks_wm_tokyo_zero_shot.npy runs/gemma_v4/ranks_hrem_tokyo_zero_shot.npy "WM vs HREM (zero)"
pt runs/gemma_wm/ranks_wm_tokyo_full.npy      runs/gemma_v4/ranks_hrem_tokyo_full.npy      "WM vs HREM (full)"
pt runs/gemma_wm/ranks_wm_tokyo_full.npy      runs/gemma_wm_incity/ranks_wm_tokyo_full.npy "WM transfer vs in-city WM (full)"
for AB in noJEPA noRetAux flat H1 noInfill; do
  pt runs/gemma_wm/ranks_wm_tokyo_zero_shot.npy runs/abl_wm_$AB/ranks_wm_tokyo_zero_shot.npy "WM vs $AB (zero)"
done
true
