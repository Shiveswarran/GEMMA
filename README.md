# GEMMA v5 — Global Embeddings for urban Mobility Modelling and Adaptation

**v5 changes (read first).** v4 evaluation had four structural caps that no
tokenizer/JEPA/AR change could fix:
1. the user's **history was never in context** (the benchmark task is
   `f(current trajectory, user history) -> rank`; LLM baselines get 15 history
   trajectories) -> v5 feeds `[hist_1] SEP ... SEP [prefix]`, the diary format
   GEMMA is pretrained on;
2. the predictor was queried at **offset 1** instead of the next position -> fixed
   (`GEMMA.encode_query`), plus an I-JEPA **cross-attention predictor** and a
   **`trail_end`/`mixed` masking** that rehearses the downstream query, with fresh
   masks every epoch (v4 reused one fixed mask per diary);
3. "few-shot" used the whole train split and adaptation was an 8-input MLP on the
   last step only (full == few) -> v5 **GEMMA-R** fine-tunes backbone + a
   realization head on every next-step pair; few-shot = <=3 trajectories/user;
4. hard top-3 activity pruning (rank=|V| outside) and a test-only POI vocabulary ->
   soft exact hierarchical path likelihood, ranking over **all** city POIs, and the
   LLM **200-user** protocol reported separately (mean+-std, 5 seeds) with three
   candidate sets: 100-nearest, 100-nearest + injected truth (headline), and
   truth + 100 random (original LLM-Move). The LLM tables rank ~101 candidates,
   NOT the city vocabulary: Istanbul LLM-Move Acc@1 0.205 exceeds the 16.4%
   100-nearest coverage, so the truth must be in the list.
5. Test metrics are split into **revisit** (target already in the user's
   history/prefix) and **explore** (new POI for that user) — heuristics score ~0
   on explore; that is where the foundation model has to earn its keep.

Pipeline: `diagnose_caps.py` (model-free ceilings) -> pretrain -> 
`train_realizer_source.py` (source-city post-training = zero-shot model) ->
`run_downstream.py` (zero/few/full, both protocols, next-POI + next-activity).
Method details: `gemma/realize.py` docstring. Run commands: `docs_RUN.md`.

---

## Background (v2-v4 design, still the backbone)

GEMMA is a JEPA-style, self-supervised **backbone** for human mobility. It is
pretrained once on the high-resource Massive-STEPS cities and then plugs into the
benchmark's downstream models (STHGCN, GETNext, LSTPM, DeepMove, Flashback, …) and
LLM baselines (LLM-Mob, LLM-ZS, LLM-Move) as a feature/KG/RAG backbone, so every
comparison is exactly **`model` vs `model + GEMMA`** under the *unchanged*
Massive-STEPS protocol (https://github.com/cruiseresearchgroup/Massive-STEPS).

Tasks: (1) next-POI recommendation, (2) next-activity prediction (new, uses the
Foursquare hierarchy), (3) trajectory classification (weekday/weekend).

## What GEMMA learns (design)
- **Latent-predictive SSL (JEPA)**: predict masked-future check-in *latents* from a
  masked diary, with an EMA target encoder. Anti-collapse regularizer is **VICReg**
  (variance–covariance) by default, **SIGReg** (LeJEPA-style) as an ablation.
- **Real Foursquare hierarchy** (`gemma/fsq.py`): top level = **activity** (~10,
  transferable), leaf = **secondary**, middle = **primary**. Joined on
  `venue_category_id`. Decoded coarse→fine from the predicted latent → the
  transferable `act@1 ≥ pri@1 ≥ sec@1` diagnostic curve. This also *is* the
  next-activity downstream task.
- **Per-user diaries**: a user's trails are joined with a learned `[SEP]` token so
  JEPA has a long-enough sequence to mask (fixes the short-trail problem).
- **Spatio-temporal encoding**: city-agnostic relative motion (log-distance +
  bearing) + multi-scale Fourier time; absolute geography stays on the downstream
  realization side.
- **POI-knowledge encoder** (`gemma/poi_encoder.py`, pluggable): `hash` (offline),
  `sbert`, or `llm_cached` (drop in precomputed embeddings from a modern LLM
  encoder — GTE/Qwen/OpenAI — no code change).
- **Motif codebook** (`MotifCodebook`): soft-assign each diary's activity-transition
  day-graph to ≥17 learnable motif prototypes (Schneider–González universality),
  optionally with **dummy HOME/WORK nodes** so a user's missing routine nodes are
  inferable from their characteristic motif — the personalization signal.
- **Geometry** (`geometry=euclidean|poincare`): Euclidean default; hyperbolic
  (Poincaré) stubbed as an ablation for the hierarchy.

## Install
```bash
pip install -r requirements.txt
```

## Data
1. **Massive-STEPS splits** under `data/<city>/<city>_checkins_{train,validation,test}.csv`
   (per-check-in rows; the repo/HF format). Pretrain uses the 6 cities with
   >~89k trails: istanbul, petaling_jaya, jakarta, new_york, kuwait_city, sao_paulo.
2. **Foursquare taxonomy CSV** (`Category ID, Category Name, Category Label`) at the
   path in `--fsq_csv`. This is the real hierarchy; `gemma/fsq.py` parses the
   `>`-separated label into activity/primary/secondary.

Offline smoke test (no data / no network):
```bash
python scripts/make_synthetic.py --out data --cities src_a src_b tgt_c
```

## Run
Pretrain the backbone (once):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --data_root data --fsq_csv personalization-apis-movement-sdk-categories.csv \
  --pretrain_cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
  --poi_encoder sbert --masking block_future --regularizer vicreg \
  --epochs 40 --batch_size 256 --d_model 256 --num_workers 8 --run_name gemma_v2
```
Reference downstream (all tasks, all cities), GEMMA frozen:
```bash
python scripts/run_downstream.py --run_dir runs/gemma_v2 --data_root data \
  --cities tokyo istanbul jakarta new_york ... \
  --tasks next_poi next_activity classification --modes frozen finetune
```
Results: `runs/gemma_v2/down_<city>_<task>_<mode>.json`, plus `pretrain.log`,
`history.jsonl`, `config.json`, `gemma.pt`.

## Integrating a SOTA downstream model (the clean comparison)
We do **not** reimplement STHGCN/GETNext/etc. — we wrap their released repos and
inject GEMMA features at one point, so their numbers are reproduced except for the
injected features. Contract (`ExternalAdapter` in `gemma/downstream.py`):
```python
adapter = ExternalAdapter(backbone, cfg, device)
poi_feat = adapter.poi_features(city)     # (n_pois, d): per-POI GEMMA feature
# In the external repo, at the line that builds its POI embedding table:
#   emb = nn.Embedding(n_pois, d0)                      # original
#   emb_table = concat([emb.weight, gemma_poi_feat])    # + GEMMA  (3-line patch)
```
Each repo gets a `adapters/<model>.py` wrapper (build order below). The LLM track
uses `adapters/rag.py`: GEMMA retrieves the user's nearest activity-motif neighbours
and their next-activity distribution, injected into the LLM-Mob/ZS/Move prompt as
retrieved context (KG/RAG), same prompting otherwise.

## Build order (recommended)
1. ✅ backbone + pretrain + reference probe + metrics + protocol (this repo).
2. Real FSQ taxonomy join + `poi_encoder=llm_cached` (precompute LLM POI embeddings).
3. Two-tower next-POI head (contrastive, same-secondary hard negatives) replacing
   the reference probe's proxy head — this is what makes next-POI numbers real.
4. `adapters/getnext.py` then `adapters/sthgcn.py` (highest-value baselines).
5. `adapters/rag.py` + LLM-Move prompt injection.
6. Ablation sweep (`experiments/matrix.md`) + transfer-gradient figures.

## Honest notes
- The reference probe's next-POI head is a *proxy* (trains the summary via an
  activity head); real next-POI ranking needs the two-tower contrastive head
  (build step 3). next-activity and classification are fully functional now.
- Hyperbolic geometry is a documented stub (`geometry=poincare`); correct
  Riemannian training is step-6 work.
- "Beating SOTA" is empirical. The design maximizes the odds (transferable
  activity grammar + motifs + modern SSL + clean adaptation) and, more importantly,
  gives a defensible *story* (the transfer-gradient + cold-start analyses in
  `experiments/matrix.md §D`) that stands even where a leaderboard cell doesn't.
EOF
