#!/usr/bin/env bash
# Headline model = the v6 HREM recipe (pre-tuning default, retriever=hrem, lr_head 2e-3, lr_bb 5e-5,
# full source queries) with 3 seeds, plus a diagnostic on the tuned WM model.
#   nohup bash scripts/run_v6seeds.sh > v6seeds.out 2>&1 &
# Stages (ONLY="D V L R"):
#   D  diagnostic: tuned WM model (gemma_final_s0) with the latent pointer / prefix channels switched
#      off at test time -> does the pointer explain the small-city loss vs v6?          ~10 min
#   V  v6 recipe, seeds 0 1 2: source training (~1.5 h each) + 9 cities zero-shot & full-shot (~40 min)
#   L  v6 recipe, seeds 0 1 2: LLM-Move protocol (gt + 100 random, last 15 trajectories)  ~5 min each
#   R  reports
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs logs
# wait for the whole run_final chain (incl. the nohup bash -c wrapper) to finish -> one job per GPU
while pgrep -f "scripts/run_final.sh" >/dev/null; do sleep 120; done
exec 9>runs/.run_final.lock; flock 9
SRC=${SRC:-"istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"}
TGT=${TGT:-"tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung"}
SEEDS=${SEEDS:-"0 1 2"}; DATA=${DATA:-data}; EXTRA=${EXTRA:-}
EV="python scripts/run_downstream.py --data_root $DATA $EXTRA"
TR="python scripts/train_realizer_source.py --data_root $DATA $EXTRA"
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }

if step D; then
  echo "=== D. pointer diagnostic on the tuned WM model ($(date)) ==="
  for DR in lptr prefix "lptr,prefix"; do
    T=_drop$(echo $DR | tr -d ',')
    $EV --run_dir runs/gemma_final_s0 --retriever wm --cities $TGT --shots zero_shot --protocols full \
      --drop_ret "$DR" --out_tag $T 2>&1 | tee logs/v6_D$T.log
  done
fi

if step V; then
  for S in $SEEDS; do
    D=runs/v6_s$S; mkdir -p $D; cp -n runs/gemma_v4/config.json runs/gemma_v4/gemma.pt $D/ 2>/dev/null
    echo "=== V. v6 recipe seed $S ($(date)) ==="
    [ -f $D/zs_hrem.pt ] || $TR --run_dir $D --cities $SRC --tune_backbone --retriever hrem --seed $S \
      2>&1 | tee logs/v6_F_s$S.log
    $EV --run_dir $D --retriever hrem --cities $TGT --shots zero_shot full --protocols full \
      --out_tag _v6s 2>&1 | tee logs/v6_E_s$S.log
  done
fi

if step L; then
  for S in $SEEDS; do
    echo "=== L. v6 recipe seed $S, LLM-Move protocol ($(date)) ==="
    $EV --run_dir runs/v6_s$S --retriever hrem --cities $TGT --shots zero_shot --protocols llm200 \
      --hist_last 15 --eval_protocol all --out_tag _v6s_llm 2>&1 | tee logs/v6_E_llm_s$S.log
  done
fi

if step R; then
  R=""; for S in $SEEDS; do R="$R v6s$S=runs/v6_s$S:hrem:_v6s"; done
  python scripts/final_report.py --baseline v6s0 --cities $TGT --shot zero_shot \
    --runs $R "wmS0=runs/gemma_final_s0:wm:_final" "wmS1=runs/gemma_final_s1:wm:_final" \
    "wmS2=runs/gemma_final_s2:wm:_final" 2>&1 | tee logs/v6_R_zero.txt
  python scripts/final_report.py --baseline v6s0 --cities $TGT --shot full --runs $R \
    "wmS0=runs/gemma_final_s0:wm:_final" 2>&1 | tee logs/v6_R_full.txt
  python scripts/final_report.py --baseline wm --cities $TGT --shot zero_shot \
    --runs "wm=runs/gemma_final_s0:wm:_final" "noLptr=runs/gemma_final_s0:wm:_droplptr" \
    "noPrefix=runs/gemma_final_s0:wm:_dropprefix" "noBoth=runs/gemma_final_s0:wm:_droplptrprefix" \
    2>&1 | tee logs/v6_R_diag.txt
fi
echo "=== DONE ($(date)) ==="
