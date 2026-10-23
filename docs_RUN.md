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

## v7.1 changes before the first HS-JEPA run (after the v6 HREM analysis)
* **Schedule**: `--min_epochs 3 --patience 4 --val_every 2000 --val_per_city 2000` for every
  world-model pretraining (v6 stopped in epoch 1: the head plateaus at once, the world model does not).
* **World-model retrieval losses** (`lambda_wm_lat`=0.5, `lambda_wm_ptr`=0.3): the predicted latent
  must (i) find the true NEW place among the user's unvisited POIs and (ii) point at the right history
  token for a return. In v6 the mixture switched the latent pointer off (lptr 0.47 -> 0.02), so the
  backbone received almost no retrieval gradient. Ablation `noRetAux` turns them off.
* **Downstream fine-tuning keeps the pretraining objective** (`lambda_wm_*` copied from `zs_wm.pt`),
  so the noJEPA ablation is noJEPA at fine-tune time too.
* **Diagnostics** in every eval line: `cond: explore@1/@10` (new target ranked among unvisited POIs),
  `revisit@1`, `lat-explore@10` (latent-only), `ptr@1`, `pi(exp/rev)` (gate calibration).
* **Per-query ranks** `ranks_<retriever>_<city>_<shot>.npy` + `scripts/paired_test.py` (paired
  bootstrap CI + McNemar). Use it for every "X helps" claim.
* `SMOKE=1 bash scripts/run_wm.sh` = 10-minute crash test before the long run.

## v8: go/no-go suite (run this BEFORE run_wm.sh)
`nohup bash scripts/run_gonogo.sh > gonogo.out 2>&1 &` (≈ one night on one RTX 6000), then
`python scripts/gonogo_report.py`. Subsets: `ONLY="G0 G2" bash scripts/run_gonogo.sh`.
* G0 zero-shot on the 9 held-out cities (pooled, paired): hand evidence / v6 HREM / no-JEPA.
* G1 value of target labels (zero / 5% / full; transfer vs in-city).
* G2 cold start: `--hist_budget 0|1|3` (only the k most recent PAST trajectories exist).
* G3 source scaling 1 -> 3 -> 6 cities, fixed recipe (`EP`, `QPC`).
* G4 backbone only: `--evidence none` removes every hand-built user/city statistic; returns must
  come from the latent pointer, new places from the latent match (mobility-CLIP: InfoNCE between
  the predicted next-place latent and text-grounded POI embeddings). Arms: JEPA-init, random-init,
  GEMMA-G (generative recipe: retrieval losses core, JEPA 0.2 auxiliary), 1 vs 6 source cities.
New flags: `--hist_budget`, `--evidence full|none`, `--out_tag` (downstream);
`--evidence` (source training, stored in zs_*.pt and re-applied downstream).

## v9: STHGCN protocol everywhere + tuned world-model pretraining (scripts/run_final.sh)
* Default evaluation = STHGCN protocol (`--eval_protocol sthgcn`): test/validation targets whose POI or
  user never occurs in the TRAIN split are not scored (Massive-STEPS STHGCN fork,
  `remove_unseen_user_poi`). `--eval_protocol all` restores the stricter view. LLM protocol unchanged.
* New pretraining objectives (world-model head): trajectory repeat-motif + activity-chain motif
  classification (`lambda_wm_motif`), time-to-next check-in (`lambda_wm_dt`), on top of hierarchical
  multi-horizon JEPA, return pointer, mobility-CLIP latent match, infill, probes, regulariser.
* `scripts/tune_pretrain.py`: leakage-free search (dev = kuwait_city VALIDATION split, pretraining
  without the dev city); Optuna TPE if installed, else seeded random search.
* `scripts/final_report.py`: per-city tables vs published STHGCN (z-test) + paired tests.
* Stages: B baselines+tuning reference, T tuning, F final pretraining (SEEDS), E held-out cities,
  S source cities full-shot, A ablations, R report.  `ONLY="B T"` etc.
