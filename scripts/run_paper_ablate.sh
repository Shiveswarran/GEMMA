#!/usr/bin/env bash
# Paper ablations on the FINAL model (GEMMA = v6 HREM, runs/v6_s{0,1,2}); all zero-shot, test-time only,
# so nothing is retrained (~2-4 min per run on one Quadro RTX 6000).
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_paper_ablate.sh > paper_ablate.out 2>&1 &
# Stages (ONLY="E G H R"):
#   E  leave-one-evidence-out: switch off one evidence family in the return / explore channel
#   G  gate: drop the exploration-law term (g_S log S_u), all personal terms, or use one global prior
#   H  history budget: only the k most recent past check-ins of each user are visible (cold start)
#   R  reports -> logs/paper_R_abl_evidence.txt, logs/paper_R_abl_gate.txt, logs/paper_R_hist.txt
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs logs
while pgrep -f "scripts/run_paper_final.sh|scripts/run_v6seeds.sh" >/dev/null; do sleep 120; done
TGT=${TGT:-"tokyo shanghai palembang melbourne sydney tangerang moscow bandung"}
SEEDS=${SEEDS:-"0 1 2"}; DATA=${DATA:-data}; EXTRA=${EXTRA:-}
EV="python scripts/run_downstream.py --data_root $DATA $EXTRA --retriever hrem --cities $TGT --shots zero_shot --protocols full"
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }

# name|flags   (names become report columns: <name>_s<seed>)
EVID=(
  "noFreq|--drop_ret hist"
  "noRecency|--drop_ret rec"
  "noTime|--drop_ret tod,dow"
  "noTrans|--drop_ret ptrans"
  "noTrip|--drop_ret prefix,is_last"
  "noRetDist|--drop_ret geo"
  "noPointer|--drop_ret lptr"
  "noActPath|--drop_exp hier"
  "noExpDist|--drop_exp geo,anchor"
  "noEvidence|--evidence none"
)
GATE=( "gateNoS|--gate_mode noS" "gateNoPersonal|--gate_mode nopersonal" "gateConst|--gate_mode const" )
HIST=( 1 2 3 5 10 20 50 )

run_set () {   # $1 = array name
  local -n A=$1
  for S in $SEEDS; do
    for spec in "${A[@]}"; do
      N=${spec%%|*}; F=${spec#*|}
      [ -f logs/abl_${N}_s$S.log ] && grep -q "SUMMARY" logs/abl_${N}_s$S.log && continue
      echo "=== $N seed $S ($(date)) ==="
      $EV --run_dir runs/v6_s$S $F --out_tag _abl_$N 2>&1 | tee logs/abl_${N}_s$S.log
    done
  done
}

step E && run_set EVID
step G && run_set GATE
if step H; then
  for S in $SEEDS; do
    for K in "${HIST[@]}"; do
      [ -f logs/abl_hist${K}_s$S.log ] && grep -q "SUMMARY" logs/abl_hist${K}_s$S.log && continue
      echo "=== history budget $K seed $S ($(date)) ==="
      $EV --run_dir runs/v6_s$S --hist_budget $K --out_tag _abl_hist$K 2>&1 | tee logs/abl_hist${K}_s$S.log
    done
  done
fi

if step R; then
  for S in $SEEDS; do
    B="full_s$S=runs/v6_s$S:hrem:_v6s"
    R=""; for spec in "${EVID[@]}"; do N=${spec%%|*}; R="$R ${N}_s$S=runs/v6_s$S:hrem:_abl_$N"; done
    python scripts/final_report.py --baseline full_s$S --cities $TGT --shot zero_shot --runs $B $R \
      2>&1 | tee logs/paper_R_abl_evidence_s$S.txt
    R=""; for spec in "${GATE[@]}"; do N=${spec%%|*}; R="$R ${N}_s$S=runs/v6_s$S:hrem:_abl_$N"; done
    python scripts/final_report.py --baseline full_s$S --cities $TGT --shot zero_shot --runs $B $R \
      2>&1 | tee logs/paper_R_abl_gate_s$S.txt
    R=""; for K in "${HIST[@]}"; do R="$R hist${K}_s$S=runs/v6_s$S:hrem:_abl_hist$K"; done
    python scripts/final_report.py --baseline full_s$S --cities $TGT --shot zero_shot --runs $B $R \
      2>&1 | tee logs/paper_R_hist_s$S.txt
  done
fi
echo "=== DONE ($(date)) ==="
