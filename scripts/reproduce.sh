#!/usr/bin/env bash
# Reproduce every result of the paper on one GPU.
#   bash scripts/reproduce.sh                 all stages
#   ONLY="eval report" bash scripts/reproduce.sh
# Stages: prep pretrain source eval llm ablate report
# Times on one 24 GB GPU: prep ~1 h, pretrain ~6 h, source ~1.5 h per seed,
# eval ~40 min per seed, llm ~5 min per seed, ablations ~3 min per run.
set -euo pipefail
DATA=${DATA:-data}
TAX=${TAX:-personalization-apis-movement-sdk-categories.csv}
RUN=${RUN:-runs/gemma}
SEEDS=${SEEDS:-"0 1 2"}
SRC="istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"
TGT="tokyo shanghai palembang melbourne sydney tangerang moscow bandung"
step () { [ -z "${ONLY:-}" ] || [[ " ${ONLY} " == *" $1 "* ]]; }
EV="python scripts/evaluate.py --run_dir $RUN --data_root $DATA --fsq_csv $TAX"

if step prep; then        # place text embeddings and semantic IDs
  python scripts/embed_pois.py --data_root $DATA --cities $SRC $TGT
  python scripts/build_semantic_ids.py --data_root $DATA --fsq_csv $TAX \
    --train_cities $SRC --all_cities $SRC $TGT
fi

if step pretrain; then    # self-supervised encoder on the six source cities
  python scripts/pretrain.py --data_root $DATA --fsq_csv $TAX --run_name $(basename $RUN) \
    --out_dir $(dirname $RUN)
fi

if step source; then      # zero-shot model: mixture trained on the source cities, one per seed
  for S in $SEEDS; do
    python scripts/train_source.py --run_dir $RUN --data_root $DATA --fsq_csv $TAX --seed $S
  done
fi

if step eval; then        # Table 2: zero-shot and fine-tuned, supervised protocol (+ source cities)
  for S in $SEEDS; do
    $EV --source $RUN/source_s$S.pt --cities $TGT --settings zero_shot finetuned --seed $S --out_tag _s$S
    $EV --source $RUN/source_s$S.pt --cities $SRC --settings finetuned --seed $S --out_tag _s$S
  done
fi

if step llm; then         # LLM-Move protocol: true place + 100 random places, last 15 trajectories
  for S in $SEEDS; do
    $EV --source $RUN/source_s$S.pt --cities $TGT --settings zero_shot --protocols llm \
      --hist_last 15 --eval_protocol all --out_tag _llm_s$S
  done
fi

if step ablate; then      # Table 5 and Figures 9-10: test-time ablations of the zero-shot model
  ABL=("noFreq|--drop_ret hist" "noRecency|--drop_ret rec" "noTime|--drop_ret tod,dow"
       "noTrans|--drop_ret ptrans" "noTrip|--drop_ret prefix,is_last" "noRetDist|--drop_ret geo"
       "noPointer|--drop_ret lptr" "noActPath|--drop_exp hier" "noExpDist|--drop_exp geo,anchor"
       "noEvidence|--evidence none" "gateNoS|--gate_mode noS" "gateNoPersonal|--gate_mode nopersonal"
       "gateConst|--gate_mode const")
  for K in 1 2 3 5 10 20 50; do ABL+=("hist$K|--hist_budget $K"); done
  for S in $SEEDS; do
    for spec in "${ABL[@]}"; do
      N=${spec%%|*}; F=${spec#*|}
      $EV --source $RUN/source_s$S.pt --cities $TGT --settings zero_shot $F --out_tag _${N}_s$S
    done
  done
  # hand-set evidence weights instead of the source-trained ones
  $EV --source none --cities $TGT --settings zero_shot --out_tag _hand
fi

if step report; then
  T=""; for S in $SEEDS; do T="$T _s$S"; done
  echo "== zero-shot, unseen cities";  python scripts/report.py --run_dir $RUN --tags $T --setting zero_shot --cities $TGT
  echo "== fine-tuned, unseen cities"; python scripts/report.py --run_dir $RUN --tags $T --setting finetuned --cities $TGT
  echo "== fine-tuned, source cities"; python scripts/report.py --run_dir $RUN --tags $T --setting finetuned --cities $SRC
  for N in noFreq noRecency noTime noTrans noTrip noRetDist noPointer noActPath noExpDist noEvidence \
           gateNoS gateNoPersonal gateConst hist1 hist5 hist50; do
    A=""; for S in $SEEDS; do A="$A _${N}_s$S"; done
    echo "== ablation $N (d = paired difference to the full model)"
    python scripts/report.py --run_dir $RUN --tags $A --baseline_tags $T --setting zero_shot --cities $TGT
  done
  echo "== exploration law"; python scripts/mobility_laws.py --data_root $DATA --cities $SRC $TGT \
    --source $RUN/source_s0.pt
  grep -h "LLM protocol" $RUN/evaluate_llm_s*.log || true
fi
