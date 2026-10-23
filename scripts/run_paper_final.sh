#!/usr/bin/env bash
# Last runs before writing (GEMMA = v6 HREM recipe, runs/v6_s{0,1,2} from run_v6seeds.sh).
#   nohup bash scripts/run_paper_final.sh > paper_final.out 2>&1 &
# Stages (ONLY="S V R"):
#   S  6 source cities, full-shot, 3 seeds (the "pretrained cities" table for v6)     ~2-3 h / seed
#   V  9 target cities, full-shot with test-time history/statistics from the TRAIN split only
#      (fairness sensitivity: default full-shot also sees the validation trajectories) ~40 min / seed
#   R  reports
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs logs
while pgrep -f "scripts/run_v7.sh|scripts/run_v6seeds.sh|scripts/run_final.sh" >/dev/null; do sleep 120; done
exec 9>runs/.run_final.lock; flock 9
SRC=${SRC:-"istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"}
TGT=${TGT:-"tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung"}
SEEDS=${SEEDS:-"0 1 2"}; DATA=${DATA:-data}; EXTRA=${EXTRA:-}
EV="python scripts/run_downstream.py --data_root $DATA $EXTRA"
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }

if step S; then
  for S in $SEEDS; do
    echo "=== S. v6 source cities full-shot seed $S ($(date)) ==="
    $EV --run_dir runs/v6_s$S --retriever hrem --cities $SRC --shots full --protocols full \
      --out_tag _v6src 2>&1 | tee logs/paper_S_s$S.log
  done
fi
if step V; then
  for S in $SEEDS; do
    echo "=== V. v6 targets full-shot, train-only visibility, seed $S ($(date)) ==="
    $EV --run_dir runs/v6_s$S --retriever hrem --cities $TGT --shots full --protocols full \
      --full_visible train --out_tag _v6trainvis 2>&1 | tee logs/paper_V_s$S.log
  done
fi
if step R; then
  R=""; T=""; for S in $SEEDS; do R="$R v6s$S=runs/v6_s$S:hrem:_v6src"; T="$T v6s$S=runs/v6_s$S:hrem:_v6trainvis"; done
  python scripts/final_report.py --cities $SRC --shot full --runs $R 2>&1 | tee logs/paper_R_src_full.txt
  python scripts/final_report.py --cities $TGT --shot full --runs $T 2>&1 | tee logs/paper_R_full_trainvis.txt
fi
echo "=== DONE ($(date)) ==="
