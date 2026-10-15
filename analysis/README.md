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
| `a8_explore_anatomy.py` | Why does exploration fail? Oracle rankers (activity/category known), bits budget, opportunity-rank law (Noulas 2012) across cities, exploration momentum | Motivation for GEMMA-X (exploration) |
| `a7_results_figures.py` | Result figures from finished runs: vs STHGCN, vs LLM-Move, history-budget curve, source scaling, revisit share vs gap | Results section |

Motivation story, revised against the real outputs of 2026-10-12 (each claim cites its evidence):
1. Places never transfer: POI vocabularies are disjoint across cities (A2, POI row; the ids in Massive-STEPS
   are city-local integers, so they must be prefixed with the city before comparing).
2. The *form* of the mobility laws transfers, their *parameters* vary: every city obeys sub-linear exploration
   (mu 0.64-0.85), a decaying novelty rate (gamma 0.17-0.50) and Zipf visitation (alpha 0.89-1.60) (A1).
3. The dominant, transferable signal is personal repetition: same-user days are 25x more similar than
   other-user days at the POI level (A3 cosine), the next place is in the user's own history for ~60%
   of STHGCN-protocol targets (A4 coverage), and user frequency alone already reaches 0.208 macro Acc@1 (A4).
4. Abstraction helps only partly: activity marginals are close across cities (JS 0.06), activity x hour
   much less so (0.24, inflated by sampling noise - read the split-half noise floor column), and the
   ORDER of activities by revisit propensity is shared (A2_revisit_rank_transfer) -> activity is the right
   unit for the return/explore gate, not for predicting the place itself.
5. GEMMA transfers a low-dimensional fusion of these personal statistics (theta); learning theta on
   source cities adds +2.8 macro points over hand-set weights, while representation pretraining adds
   nothing (final_R_ablation) - see A5 and A7.
