#!/usr/bin/env bash
# GEMMA go/no-go suite: four cheap, decisive questions BEFORE any long pretraining run.
#   nohup bash scripts/run_gonogo.sh > gonogo.out 2>&1 &   ;   tail -f gonogo.out
#   then:  python scripts/gonogo_report.py            (prints the decision table)
#
#   G0  zero-shot on all 9 held-out cities: hand evidence only / v6 HREM / no-JEPA HREM
#   G1  value of TARGET LABELS (zero / 5% / full, transfer vs in-city) -> how much room
#       any pretraining has on this metric
#   G2  cold start: history budget 0 / 1 / 3 past trajectories -> where pretraining matters
#   G3  source scaling: 1 -> 3 -> 6 source cities (fixed recipe) -> does more data help?
#   G4  "strong backbone" test: NO hand-built evidence; the backbone must find returns
#       (latent pointer) and new places (latent match) itself; JEPA-init vs random vs
#       generative (GEMMA-G) pretraining, and 1 vs 6 source cities
# Runs on GPU 0 only, one job at a time. Every eval writes per-query ranks for paired tests.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/scratch/umni5/a/${USER:-$(whoami)}/hf}; export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
SRC=${SRC:-"istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"}
TGT=${TGT:-"tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung"}
LV=${LV_CITIES:-"tokyo shanghai melbourne sydney moscow"}     # label-value subset (G1)
read -r -a _S <<< "$SRC"; S1="${_S[0]}"; S3="${_S[*]:0:3}"     # 1- and 3-city source subsets
QPC=${QPC:-50000}; EP=${EP:-2}; VPC=${VPC:-1000}; DATA=${DATA:-data}; EXTRA=${EXTRA:-}
FIX="--epochs $EP --min_epochs $EP --patience 99 --max_queries_per_city $QPC --batch 128 --val_per_city $VPC $EXTRA"
EV="python scripts/run_downstream.py --data_root $DATA --protocols full $EXTRA"
TR="python scripts/train_realizer_source.py --data_root $DATA"
mkdir -p logs

[ -n "${SKIP_GPU_CHECK:-}" ] || python - <<'PY' || { echo "No usable GPU -> stopping."; exit 1; }
import torch, sys
ok = torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0) if ok else "NONE")
sys.exit(0 if ok else 1)
PY
for d in runs/gemma_v4/zs_hrem.pt runs/abl_random/zs_hrem.pt runs/abl_incity/gemma.pt; do
  [ -f "$d" ] || { echo "missing $d (run scripts/run_hrem.sh first)"; exit 1; }
done
newdir () { mkdir -p "$1" && cp runs/gemma_v4/config.json runs/gemma_v4/gemma.pt "$1"/; }
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }   # ONLY="G2 G3" to run a subset

if step G0; then
echo "=== G0 zero-shot, 9 held-out cities ($(date)) ==="
# hand evidence = HREM with its hand-set weights, no source training (v4 backbone for activity)
$EV --run_dir runs/abl_incity --retriever hrem --cities $TGT --shots zero_shot \
  --out_tag _G0 2>&1 | tee logs/G0_hand.log
$EV --run_dir runs/gemma_v4  --retriever hrem --cities $TGT --shots zero_shot --out_tag _G0 2>&1 | tee logs/G0_v6.log
$EV --run_dir runs/abl_random --retriever hrem --random_backbone --cities $TGT --shots zero_shot \
  --out_tag _G0 2>&1 | tee logs/G0_nojepa.log
fi

if step G1; then
echo "=== G1 label value: zero / 5% / full, transfer vs in-city ($(date)) ==="
$EV --run_dir runs/gemma_v4 --retriever hrem --cities $LV --shots few_shot full --few_shot_frac 0.05 \
  --out_tag _G1 2>&1 | tee logs/G1_transfer.log
$EV --run_dir runs/abl_incity --retriever hrem --random_backbone --no_init_from_zs --cities $LV \
  --shots few_shot full --few_shot_frac 0.05 --out_tag _G1 2>&1 | tee logs/G1_incity.log
fi

if step G2; then
echo "=== G2 cold start: history budget 0 / 1 / 3 ($(date)) ==="
for K in 0 1 3; do
  $EV --run_dir runs/abl_incity --retriever hrem --cities $TGT --shots zero_shot \
    --hist_budget $K --out_tag _G2h$K 2>&1 | tee logs/G2_hand_h$K.log
  $EV --run_dir runs/gemma_v4  --retriever hrem --cities $TGT --shots zero_shot \
    --hist_budget $K --out_tag _G2h$K 2>&1 | tee logs/G2_v6_h$K.log
  $EV --run_dir runs/abl_random --retriever hrem --random_backbone --cities $TGT --shots zero_shot \
    --hist_budget $K --out_tag _G2h$K 2>&1 | tee logs/G2_nojepa_h$K.log
done
fi

if step G3; then
echo "=== G3 source scaling 1 -> 3 -> 6 cities (with evidence) ($(date)) ==="
for S in "1:$S1" "3:$S3" "6:$SRC"; do
  N=${S%%:*}; C=${S#*:}
  newdir runs/scale_$N
  $TR --run_dir runs/scale_$N --cities $C \
    --retriever hrem --tune_backbone $FIX 2>&1 | tee logs/G3_train_$N.log
  $EV --run_dir runs/scale_$N --retriever hrem --cities $TGT --shots zero_shot --out_tag _G3 \
    2>&1 | tee logs/G3_eval_$N.log
done
fi

if step G4; then
echo "=== G4 backbone-only (no hand evidence) ($(date)) ==="
# a) JEPA-v4 init + supervised source training      b) random init + same
# c) GEMMA-G: random init, generative world-model recipe (latent-pointer + mobility-CLIP
#    retrieval losses as the core, JEPA as a 0.2 auxiliary)    d) = a) but 1 source city
for S in "bo_jepa:hrem::$SRC" "bo_rand:hrem:--random_backbone --lr_backbone 3e-4 --lr_head 1e-3:$SRC" \
         "bo_gen:wm:--random_backbone --lambda_wm_jepa 0.2 --lambda_wm_infill 0.1 --lambda_wm_lat 1.0 --lambda_wm_ptr 1.0 --lr_backbone 3e-4 --lr_head 1e-3:$SRC" \
         "bo_jepa1:hrem::$S1"; do
  IFS=: read -r NAME RET FLAGS CITIES <<< "$S"
  newdir runs/$NAME
  $TR --run_dir runs/$NAME --cities $CITIES \
    --retriever $RET --tune_backbone --evidence none $FIX $FLAGS 2>&1 | tee logs/G4_train_$NAME.log
  RB=""; [[ "$FLAGS" == *random_backbone* ]] && RB="--random_backbone"
  $EV --run_dir runs/$NAME --retriever $RET $RB --cities $TGT --shots zero_shot --out_tag _G4 \
    2>&1 | tee logs/G4_eval_$NAME.log
done
fi

echo "=== DONE ($(date)) ==="
python scripts/gonogo_report.py --tgt "$TGT" --lv "$LV" | tee logs/gonogo_report.txt
