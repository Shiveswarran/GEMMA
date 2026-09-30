# GEMMA v5 — run procedure (cluster)

## 0. Environment (all caches off the home quota)
```bash
cd /scratch/umni5/a/shives/POI_RecSys/GEMMA          # repo root (relative paths!)
conda activate gemma
export HF_HOME=/scratch/umni5/a/$USER/hf HF_HUB_CACHE=/scratch/umni5/a/$USER/hf/hub HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CSV=fsk_categories.csv
SRC="istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo"
ALL="tokyo istanbul jakarta new_york kuwait_city petaling_jaya sao_paulo bandung beijing melbourne moscow palembang shanghai sydney tangerang"
```

## 1. Model-free diagnostic (CPU, run first; minutes per city)
```bash
python scripts/diagnose_caps.py --data_root data --cities $ALL --out diag_caps.json 2>&1 | tee diag_caps.log
```
Coverage of history / 100-nearest / Markov signals and 4 heuristic rankers under
both protocols. If `combo` is close to SOTA, the task is mostly history +
transitions + geography, and GEMMA must beat `combo` before anything else.

## 2. v5 downstream on the EXISTING gemma_v4 backbone (no re-pretraining)
Uses the pooled predictor automatically (old config has no `predictor_type`).
```bash
# 2a. source-city post-training of the realizer (zero-shot model)
python scripts/train_realizer_source.py --run_dir runs/gemma_v4 --data_root data \
    --cities $SRC --tune_backbone 2>&1 | tee zs_v4.log
# 2b. quick check on one city first
python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data --cities tokyo \
    --shots zero_shot few_shot full --protocols full llm200 2>&1 | tee eval_v5_tokyo.log
# 2c. all 15 cities
python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data --cities $ALL \
    --shots zero_shot few_shot full --protocols full llm200 2>&1 | tee eval_v5_on_v4.log
```
Outputs: `runs/gemma_v4/v5_<city>_<shot>.json`, `benchmark_summary_v5.json`, and a
summary table with STHGCN (few/full) and best LLM-Move (zero-shot) references.
OOM on a big city -> `--ft_batch 64`.  Slow epochs -> `--ft_max_queries 200000`.

## 2d. Seeds (for the paper: mean +- std; a 0.006 margin on 1,097 Tokyo queries is inside 1 SE ~ 0.013)
```bash
for S in 1 2 3; do python scripts/run_downstream.py --run_dir runs/gemma_v4 --data_root data \
  --cities tokyo --shots zero_shot few_shot full --seed $S 2>&1 | tee eval_v5_tokyo_s$S.log
  mkdir -p runs/gemma_v4/seed$S && cp runs/gemma_v4/v5_tokyo_*.json runs/gemma_v4/seed$S/; done
```

## 3. Re-pretrain the v5 backbone (xattn predictor + mixed masking), reuse sid_cache
```bash
python scripts/run_pretrain.py --data_root data --fsq_csv $CSV --pretrain_cities $SRC \
    --poi_encoder llm_cached --leaf_head ar --predictor_type xattn --masking mixed \
    --epochs 40 --batch_size 256 --num_workers 8 --sid_cache sid_cache \
    --run_name gemma_v5 2>&1 | tee pretrain_v5.log
```
Then repeat step 2 with `--run_dir runs/gemma_v5`.  v4-vs-v5 backbone on the same
downstream is itself an ablation row (predictor + masking).

## 4. Ablations (downstream flags; each writes to the same run_dir, so copy the
summary json between runs or use a separate --run_dir copy)
| flag | question |
|---|---|
| `--no_history` | value of history-in-context (v4 behaviour) |
| `--rerank_m 0` | value of the AR semantic-ID rerank |
| `--ft_lr_backbone 0` | frozen backbone vs fine-tuned |
| `--last_position_only` | every next-step pair vs last only |
| `--no_init_from_zs` | source post-training as initialisation |
| `--causal_history` | only history trajectories that start earlier (stricter) |
| `--few_shot_k 1/3/5` | few-shot curve |
| no `zs_realizer.pt` | zero-shot with hand-initialised fusion only |
