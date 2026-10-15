#!/usr/bin/env bash
# GEMMA analysis suite -> analysis_out/{tables,figures}.  Run from the repo root.
#   A1-A4, A7, A8 are CPU-only (safe to run while tuning occupies the GPU).
#   A5, A6(a) load the trained model; they use the GPU briefly (or add EXTRA=--allow_cpu).
#   ONLY="A1 A2" bash analysis/run_all.sh  to run a subset.
set -uo pipefail
RUN=${RUN:-runs/gemma_v4}; DATA=${DATA:-data}; OUT=${OUT:-analysis_out}; EXTRA=${EXTRA:-}
TGT="tokyo beijing shanghai palembang melbourne sydney tangerang moscow bandung"
step () { [ "${ONLY:-}" = "" ] || [[ " ${ONLY} " == *" $1 "* ]]; }
mkdir -p $OUT logs
C="--run_dir $RUN --data_root $DATA --out $OUT"
step A1 && python analysis/a1_mobility_laws.py $C 2>&1 | tee logs/analysis_A1.log
step A2 && python analysis/a2_activity_patterns.py $C 2>&1 | tee logs/analysis_A2.log
step A3 && python analysis/a3_daily_sequences_pca.py $C 2>&1 | tee logs/analysis_A3.log
step A4 && python analysis/a4_information_value.py $C --cities $TGT 2>&1 | tee logs/analysis_A4.log
step A8 && python analysis/a8_explore_anatomy.py $C --cities $TGT 2>&1 | tee logs/analysis_A8.log
step A7 && python analysis/a7_results_figures.py $C --cities $TGT 2>&1 | tee logs/analysis_A7.log
step A5 && CUDA_VISIBLE_DEVICES=${GPU:-0} python analysis/a5_learned_behaviour.py $C --compare_run runs/abl_random \
  --cities tokyo sydney moscow --probe_src istanbul new_york $EXTRA 2>&1 | tee logs/analysis_A5.log
step A6 && CUDA_VISIBLE_DEVICES=${GPU:-0} python analysis/a6_cover_example.py $C --city tokyo \
  --pair_cities istanbul tokyo $EXTRA 2>&1 | tee logs/analysis_A6.log
echo "tables: $OUT/tables   figures: $OUT/figures"
