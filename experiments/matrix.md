# GEMMA v2 — experiment & ablation matrix (paper grid)

Every experiment is a `Config` diff; the runner sweeps these axes. Pretrain ONCE
per backbone config on the 6 source cities (>~89k trails), evaluate on all 15.

## A. Headline comparison (per city × task)
For each downstream model D ∈ {STHGCN, GETNext, LSTPM, DeepMove, Flashback,
Graph-Flashback, LoTNext, probe} and each task T ∈ {next_poi, next_activity,
classification}: report `D` vs `D + GEMMA` (frozen / finetune / prefix).
LLM track: {LLM-Mob, LLM-ZS, LLM-Move} vs same + GEMMA-RAG.

## B. Backbone ablations (pretrain axis) — measure act@1/pri@1/sec@1 + downstream lift
1. SSL objective:      JEPA(latent) | +hierarchy | +motif | +reg    (drop-one)
2. Masking:            block_future | random_span | activity_block | causal_next
3. Regularizer:        vicreg | sigreg | none
4. Geometry:           euclidean | poincare (hyperbolic)
5. POI encoder:        hash | sbert | llm_cached
6. Motif:              off | on | on+dummy_home_work
7. Diary:              per-trail | per-user-diary([SEP]) ; max_diary_len ∈ {64,128,256}
8. Source cities:      single | 3 | 6 ; pretrain size scaling

## C. Adaptation ablations (downstream axis)
frozen features | finetune | prefix-token | LoRA-on-backbone

## D. Transfer analyses (the paper's story, not just leaderboard cells)
- per-level curve: act@1 ≥ pri@1 ≥ sec@1 ≥ poi@1 (invariant→specific gradient)
- cold-start / new-venue slice of predictions_*.csv
- zero-shot (source only) → few-shot → in-city ladder
- cross-city motif-distribution similarity (universality evidence)

## E. Extra downstream tasks to add
- next-category-at-each-level (activity/primary/secondary) — 3 heads
- trajectory user-identification (who) — probe over user bank
- next-POI time / dwell regression (auxiliary)
- weekday/weekend + time-of-day-bucket classification (Massive-STEPS §4.3)

## v5 downstream ablations (run_downstream.py flags)
- context: full history (default) | `--no_history` | `--causal_history` | `--hist_trails 5/15/30`
- realization: `--rerank_m 0/50/100/200` | `--ft_lr_backbone 0/1e-4/3e-4` | `--last_position_only`
- zero-shot: source realizer with/without `--tune_backbone` | none (hand fusion)
- few-shot curve: `--few_shot_k 1 3 5 10`
- backbone: gemma_v4 (pooled, block_future) vs gemma_v5 (xattn, mixed)
