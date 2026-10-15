#!/usr/bin/env bash
# GEMMA v7 = v6 HREM + learned population evidence (both channels) + opportunity-rank explore
#            + momentum / far-from-home explore gate.  Same backbone (gemma_v4), same recipe as v6.
#   nohup bash scripts/run_v7.sh > v7.out 2>&1 &
# Stages (ONLY="T E S A N R"; SEEDS="0 1 2"):
#   T  source training, 6 source cities, WITH their own statistics (leave-one-out) and stat
#      dropout 0.5 -> one head that works with and without a city log           ~2-3 h / seed
#   E  9 held-out cities: zero_shot (strict), zero_shot_u (unlabelled target log, NO training),
#      full (fine-tune on target train), + LLM-Move protocol (strict zero-shot)  ~1 h / seed
#   S  6 source cities: full-shot and zero_shot_u (in-domain)                     ~2-3 h / seed
#   A  test-time ablations on seed 0 (no retraining): each new evidence group switched off
#   N  train-time ablation: v7 head trained WITHOUT source statistics (weights of the
#      population evidence stay hand-set, as in v6)                              ~2 h
#   R  reports (paired bootstrap vs v6 seed 0, z-test vs published STHGCN)
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs logs
while pgrep -f "scripts/run_v6seeds.sh|scripts/run_final.sh" >/dev/null; do sleep 120; done
exec 9>runs/.run_final.lock; flock 9
SRC=${SRC:-"istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"}
TGT=${TGT:-"tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung"}
SEEDS=${SEEDS:-"0 1 2"}; DATA=${DATA:-data}; EXTRA=${EXTRA:-}
EV="python scripts/run_downstream.py --data_root $DATA $EXTRA"
TR="python scripts/train_realizer_source.py --data_root $DATA $EXTRA"
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }
newdir () { mkdir -p "$1"; cp -n runs/gemma_v4/config.json runs/gemma_v4/gemma.pt "$1"/ 2>/dev/null; }

if step T; then
  for S in $SEEDS; do
    D=runs/v7_s$S; newdir $D
    echo "=== T. v7 source training seed $S ($(date)) ==="
    [ -f $D/zs_hrem7.pt ] || $TR --run_dir $D --cities $SRC --tune_backbone --retriever hrem7 \
      --collab --stat_dropout 0.5 --seed $S 2>&1 | tee logs/v7_T_s$S.log
  done
fi

if step E; then
  for S in $SEEDS; do
    D=runs/v7_s$S
    echo "=== E. v7 held-out cities seed $S ($(date)) ==="
    $EV --run_dir $D --retriever hrem7 --cities $TGT --shots zero_shot zero_shot_u full --protocols full \
      --out_tag _v7 2>&1 | tee logs/v7_E_s$S.log
    $EV --run_dir $D --retriever hrem7 --cities $TGT --shots zero_shot --protocols llm200 --hist_last 15 \
      --eval_protocol all --out_tag _v7_llm 2>&1 | tee logs/v7_E_llm_s$S.log
  done
fi

if step S; then
  for S in $SEEDS; do
    D=runs/v7_s$S
    echo "=== S. v7 source cities seed $S ($(date)) ==="
    $EV --run_dir $D --retriever hrem7 --cities $SRC --shots zero_shot_u full --protocols full \
      --out_tag _v7src 2>&1 | tee logs/v7_S_s$S.log
  done
fi

if step A; then
  D=runs/v7_s0
  echo "=== A. v7 test-time ablations ($(date)) ==="
  for AB in "noPopRet:--drop_ret ctr,icf" "noPopExp:--drop_exp ctr,ccat,icf" \
            "noICF:--drop_ret icf --drop_exp icf" "noOrank:--drop_exp orank_sec,orank_act"; do
    N=${AB%%:*}; F=${AB#*:}
    $EV --run_dir $D --retriever hrem7 --cities $TGT --shots zero_shot_u --protocols full $F \
      --out_tag _v7_$N 2>&1 | tee logs/v7_A_$N.log
  done
  $EV --run_dir $D --retriever hrem7 --cities $TGT --shots zero_shot --protocols full \
    --drop_exp orank_sec,orank_act --out_tag _v7_noOrank 2>&1 | tee -a logs/v7_A_noOrank.log
fi

if step N; then
  D=runs/v7_nostat; newdir $D
  echo "=== N. v7 head without source statistics ($(date)) ==="
  [ -f $D/zs_hrem7.pt ] || $TR --run_dir $D --cities $SRC --tune_backbone --retriever hrem7 --seed 0 \
    2>&1 | tee logs/v7_N.log
  # --collab 1: the city statistics ARE given at test time; only their weights were never learned
  $EV --run_dir $D --retriever hrem7 --cities $TGT --shots zero_shot zero_shot_u full --protocols full \
    --collab 1 --out_tag _v7nostat 2>&1 | tee -a logs/v7_N.log
fi

if step R; then
  R=""; for S in $SEEDS; do R="$R v7s$S=runs/v7_s$S:hrem7:_v7"; done
  for SH in zero_shot full; do      # paired vs the v6 recipe, same shot
    python scripts/final_report.py --baseline v6s0 --cities $TGT --shot $SH \
      --runs "v6s0=runs/v6_s0:hrem:_v6s" $R 2>&1 | tee logs/v7_R_$SH.txt
  done
  python scripts/final_report.py --cities $TGT --shot zero_shot_u --runs $R 2>&1 | tee logs/v7_R_zero_shot_u.txt
  R=""; for S in $SEEDS; do R="$R v7s$S=runs/v7_s$S:hrem7:_v7src"; done
  for SH in zero_shot_u full; do
    python scripts/final_report.py --cities $SRC --shot $SH --runs $R 2>&1 | tee logs/v7_R_src_$SH.txt
  done
  A="v7=runs/v7_s0:hrem7:_v7"
  for N in noPopRet noPopExp noICF noOrank; do A="$A $N=runs/v7_s0:hrem7:_v7_$N"; done
  python scripts/final_report.py --baseline v7 --cities $TGT --shot zero_shot_u --runs $A \
    2>&1 | tee logs/v7_R_ablation_u.txt
  python scripts/final_report.py --baseline v7 --cities $TGT --shot zero_shot \
    --runs "v7=runs/v7_s0:hrem7:_v7" "noOrank=runs/v7_s0:hrem7:_v7_noOrank" 2>&1 | tee logs/v7_R_ablation_zs.txt
  for SH in zero_shot zero_shot_u full; do
    python scripts/final_report.py --baseline v7 --cities $TGT --shot $SH \
      --runs "v7=runs/v7_s0:hrem7:_v7" "nostat=runs/v7_nostat:hrem7:_v7nostat" 2>&1 | tee logs/v7_R_nostat_$SH.txt
  done
fi
echo "=== DONE ($(date)) ==="
