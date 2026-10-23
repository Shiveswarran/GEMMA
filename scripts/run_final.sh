#!/usr/bin/env bash
# GEMMA conference pipeline under the STHGCN test protocol (default everywhere now).
#   nohup bash scripts/run_final.sh > final.out 2>&1 &      (a lock file prevents two copies)
# Stages (select with ONLY="B T F E S A R"):
#   B  baselines re-scored on the STHGCN protocol (v6 HREM, hand-set)            ~40 min
#   T  hyper-parameter search (dev = kuwait_city validation split, no test data)  TRIALS x ~35 min
#   F  final pretraining on all 6 source cities with the best flags, SEEDS        ~3-4 h / seed
#   E  9 held-out cities: zero-shot (+ LLM protocol), few-shot 5%, full-shot      ~2 h / seed
#   S  6 source cities: full-shot (in-domain; not zero-shot)                      ~2-3 h
#   A  ablations at the tuning budget, zero-shot on the 9 held-out cities         ~45 min each
#   R  report tables
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-/scratch/umni5/a/${USER:-$(whoami)}/hf}; export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p runs; exec 9>runs/.run_final.lock
flock -n 9 || { echo "another run_final.sh is running (runs/.run_final.lock) -> exit"; exit 1; }
SRC=${SRC:-"istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"}
TGT=${TGT:-"tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung"}
DEV=${DEV:-kuwait_city}; TRIALS=${TRIALS:-12}; EP_T=${EP_T:-2}; QPC_T=${QPC_T:-25000}
EPOCHS=${EPOCHS:-3}; QPC=${QPC:-100000}; SEEDS=${SEEDS:-"0"}
DATA=${DATA:-data}; EXTRA=${EXTRA:-}
EV="python scripts/run_downstream.py --data_root $DATA $EXTRA"
TR="python scripts/train_realizer_source.py --data_root $DATA $EXTRA"
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }
newdir () { mkdir -p "$1" && cp runs/gemma_v4/config.json runs/gemma_v4/gemma.pt "$1"/; }
mkdir -p logs

# default flags if no tuning result exists (design defaults)
DEFAULT="--lr_backbone 3e-4 --lr_head 1e-3 --lambda_wm_jepa 1.0 --lambda_wm_infill 0.5 --lambda_wm_ptr 1.0 \
 --lambda_wm_lat 0.5 --lambda_wm_motif 0.3 --lambda_wm_dt 0.1 --wm_H 3 --wm_infill 0.15 --ema_decay 0.996 --random_backbone"
best_flags () { [ -f runs/tune/best.json ] && python -c "import json;print(' '.join(json.load(open('runs/tune/best.json'))['flags']))" || echo "$DEFAULT"; }

if step B; then
  echo "=== B. baselines on the STHGCN protocol ($(date)) ==="
  $EV --run_dir runs/gemma_v4 --retriever hrem --cities $TGT --shots zero_shot --protocols full --out_tag _v6 \
    2>&1 | tee logs/final_B_v6.log
  $EV --run_dir runs/abl_incity --retriever hrem --cities $TGT --shots zero_shot --protocols full --out_tag _hand \
    2>&1 | tee logs/final_B_hand.log
  # reference for the tuning leaderboard: the v6 HREM recipe at the SAME tuning budget, trained
  # without the dev city and scored on its validation split -- every trial must beat this
  SRC_NODEV=$(for c in $SRC; do [[ " $DEV " == *" $c "* ]] || echo -n "$c "; done)
  newdir runs/tune_ref_hrem
  $TR --run_dir runs/tune_ref_hrem --cities $SRC_NODEV --retriever hrem --tune_backbone \
    --epochs $EP_T --min_epochs $EP_T --patience 99 --max_queries_per_city $QPC_T --batch 128 \
    --val_per_city 1000 2>&1 | tee logs/final_B_ref.log
  $EV --run_dir runs/tune_ref_hrem --retriever hrem --cities $DEV --shots zero_shot --protocols full \
    --eval_split validation --out_tag _tune 2>&1 | tee -a logs/final_B_ref.log
  python - "$DEV" <<'PY' | tee runs/tune_ref_hrem/objective.txt
import json, sys
devs = sys.argv[1].split(); o = []
for c in devs:
    m = json.load(open(f"runs/tune_ref_hrem/hrem_{c}_zero_shot_tune.json")); o.append(m["acc@1"] + 0.25 * m["ndcg@5"])
    print(f"reference v6-HREM on {c} (validation): acc@1={m['acc@1']:.4f} ndcg@5={m['ndcg@5']:.4f}")
print(f"REFERENCE OBJECTIVE = {sum(o) / len(o):.4f}  (tuning trials must exceed this)")
PY
fi

if step T; then
  echo "=== T. hyper-parameter search ($(date)) ==="
  python scripts/tune_pretrain.py --base_run runs/gemma_v4 --data_root $DATA --sources $SRC --dev $DEV \
    --trials $TRIALS --epochs $EP_T --qpc $QPC_T ${EXTRA:+--extra="$EXTRA"} 2>&1 | tee -a logs/final_T_tune.log
fi
FLAGS=$(best_flags); echo "flags: $FLAGS"

if step F; then
  for S in $SEEDS; do
    echo "=== F. final pretraining seed $S ($(date)) ==="
    newdir runs/gemma_final_s$S
    $TR --run_dir runs/gemma_final_s$S --cities $SRC --retriever wm --tune_backbone $FLAGS --seed $S \
      --epochs $EPOCHS --max_queries_per_city $QPC --batch 128 --min_epochs 2 --patience 4 \
      --val_every 2000 --val_per_city 2000 2>&1 | tee logs/final_F_s$S.log
  done
fi

if step E; then
  for S in $SEEDS; do
    echo "=== E. held-out cities, seed $S ($(date)) ==="
    $EV --run_dir runs/gemma_final_s$S --retriever wm --random_backbone --cities $TGT \
      --shots zero_shot few_shot full --few_shot_frac 0.05 --protocols full --out_tag _final 2>&1 | tee logs/final_E_s$S.log
    $EV --run_dir runs/gemma_final_s$S --retriever wm --random_backbone --cities $TGT --shots zero_shot \
      --protocols llm200 --hist_last 15 --eval_protocol all --out_tag _final_llm 2>&1 | tee logs/final_E_llm_s$S.log
  done
fi

if step S; then
  for S in $SEEDS; do
    echo "=== S. source cities full-shot, seed $S ($(date)) ==="
    $EV --run_dir runs/gemma_final_s$S --retriever wm --random_backbone --cities $SRC --shots full \
      --protocols full --out_tag _final_src 2>&1 | tee logs/final_S_s$S.log
  done
fi

if step A; then
  echo "=== A. ablations at the tuning budget ($(date)) ==="
  INIT_TOGGLE=$(echo "$FLAGS" | grep -q -- "--random_backbone" && echo "jepaInit" || echo "randomInit")
  BASEF=$(echo "$FLAGS" | sed 's/--random_backbone//')
  RB=$(echo "$FLAGS" | grep -q -- "--random_backbone" && echo "--random_backbone" || echo "")
  # ABL="full noWM ..." selects a subset (order = priority).  Later flags override $BASEF.
  ALL_ABL=("full:" "noWM:--lambda_wm_jepa 0 --lambda_wm_infill 0 --lambda_wm_ptr 0 --lambda_wm_lat 0 --lambda_wm_motif 0 --lambda_wm_dt 0" \
           "noJEPA:--lambda_wm_jepa 0 --lambda_wm_infill 0" "noPtr:--lambda_wm_ptr 0" "noLat:--lambda_wm_lat 0" \
           "noDT:--lambda_wm_dt 0" "addMotif:--lambda_wm_motif 0.3" "flat:--wm_flat" "H1:--wm_H 1" "$INIT_TOGGLE:TOGGLE")
  for AB in "${ALL_ABL[@]}"; do
    NAME=${AB%%:*}
    [ -z "${ABL:-}" ] || [[ " ${ABL} " == *" $NAME "* ]] || continue
    [ -f runs/abl_final_$NAME/wm_tokyo_zero_shot_abl.json ] && [ -f runs/abl_final_$NAME/wm_bandung_zero_shot_abl.json ] \
      && { echo "skip $NAME (done)"; continue; }
    F2=${AB#*:}
    if [ "$F2" = "TOGGLE" ]; then
      F2=""; RBA=$([ -n "$RB" ] && echo "" || echo "--random_backbone")
    else RBA=$RB; fi
    newdir runs/abl_final_$NAME
    $TR --run_dir runs/abl_final_$NAME --cities $SRC --retriever wm --tune_backbone $BASEF $RBA $F2 \
      --epochs $EP_T --min_epochs $EP_T --patience 99 --max_queries_per_city $QPC_T --batch 128 \
      --val_per_city 1000 2>&1 | tee logs/final_A_$NAME.log
    $EV --run_dir runs/abl_final_$NAME --retriever wm --random_backbone --cities $TGT --shots zero_shot \
      --protocols full --out_tag _abl 2>&1 | tee -a logs/final_A_$NAME.log
  done
fi

if step R; then
  echo "=== R. report ($(date)) ==="
  S0=$(echo $SEEDS | awk '{print $1}')
  python scripts/final_report.py --baseline v6 --cities $TGT --shot zero_shot \
    --runs "v6=runs/gemma_v4:hrem:_v6" "hand=runs/abl_incity:hrem:_hand" "GEMMA=runs/gemma_final_s$S0:wm:_final" \
    2>&1 | tee logs/final_R_zero.txt
  for SH in few_shot full; do
    python scripts/final_report.py --cities $TGT --shot $SH --runs "GEMMA=runs/gemma_final_s$S0:wm:_final" \
      2>&1 | tee logs/final_R_$SH.txt
  done
  python scripts/final_report.py --cities $SRC --shot full --runs "GEMMA=runs/gemma_final_s$S0:wm:_final_src" \
    2>&1 | tee logs/final_R_source_full.txt
  AR=""; for N in full noWM noJEPA noPtr noLat noDT addMotif flat H1 jepaInit randomInit; do
    [ -d runs/abl_final_$N ] && AR="$AR $N=runs/abl_final_$N:wm:_abl"; done
  [ -n "$AR" ] && python scripts/final_report.py --baseline full --cities $TGT --shot zero_shot --runs $AR \
    2>&1 | tee logs/final_R_ablation.txt
fi
echo "=== DONE ($(date)) ==="
