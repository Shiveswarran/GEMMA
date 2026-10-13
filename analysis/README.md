# GEMMA analysis suite (problem statement, motivation, interpretability)

Run from the repo root: `bash analysis/run_all.sh` (or `ONLY="A1 A3" bash analysis/run_all.sh`).
Outputs: `analysis_out/tables/*.csv|.tex` (LaTeX booktabs) and `analysis_out/figures/*.pdf|.png`.

| Script | Question | Paper use |
|---|---|---|
| `a1_mobility_laws.py` | Are the laws that decide the next place the same in every city? (revisit share, exploration law S~n^μ, novelty law P(new)~S^-γ, visitation Zipf α, unseen-target share) | Problem statement: "places differ, laws do not" |
| `a2_activity_patterns.py` | What is the transferable unit? Cross-city divergence along the abstraction ladder POI → category → activity → activity×hour → activity transitions; revisit rate by activity | Motivation for activity tokens + activity-gated return/explore |
| `a3_daily_sequences_pca.py` | Why activity-based pretraining? PCA rank of user-day matrices, cosine similarity of user-days (same user / other users / other city) at POI vs category vs activity level, source-PC subspace transfer, eigen-days | Advisor's PCA + cosine analysis |
| `a4_information_value.py` | Which information predicts the next place, and does GEMMA use it? One-signal rankers + coverage on the STHGCN protocol | "What information is useful" table |
| `a5_learned_behaviour.py` | What did GEMMA learn? Learned evidence exponents, gate calibration by activity, learned EPR novelty curve, linear probes of the backbone latent (JEPA vs no JEPA), PCA of the latent | Interpretability section |
| `a6_cover_example.py` | Real examples for Fig. 1: a correct zero-shot prediction in Tokyo with its history and evidence; two real user-days from different cities with the same activity pattern and no shared place | Cover figure |
| `a7_results_figures.py` | Result figures from finished runs: vs STHGCN, vs LLM-Move, history-budget curve, source scaling, revisit share vs gap | Results section |

The suggested motivation story (each claim backed by one output):
1. Cities share no places (A2: 0 shared POIs) yet their mobility laws are nearly identical (A1).
2. Abstraction makes cities similar: activity rhythms diverge far less than POIs or categories (A2a), and
   activity-days are low-rank, consistent within a user and shared across cities (A3).
3. Most predictive information sits in the user's own history plus geometry and activity (A4) —
   all available in a new city without training.
4. GEMMA learns exactly these laws on source cities (A5) and applies them zero-shot (A6, A7).

Privacy: A6 shifts coordinates to the current location, jitters them (±200 m) and hides POI names and user ids.
Check the Massive-STEPS / Foursquare licence before publishing real trajectories.
