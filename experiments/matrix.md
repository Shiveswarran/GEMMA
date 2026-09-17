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
