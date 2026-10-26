# paper/ — figures, tables and diagrams for the GEMMA paper

Everything here is regenerated from the experiment logs, so the numbers in the text, the tables and the figures
always agree. GEMMA = the v6 HREM recipe, seeds 0/1/2 (`runs/v6_s{0,1,2}`).

```
paper/
├── run_all.sh            one command: collect → figures → tables → architecture PDF → preview PDF
├── collect.py            logs/*.log, logs/*_R_*.txt, analysis_out/tables/*.csv  →  data/results.json
├── data_api.py           every aggregate (macro, pooled, 95% CI, seed sd) is computed here, once
├── baselines.py          Massive-STEPS Table 4 (11 supervised models) and Table 5 (15 LLM agents), per city
├── style.py              sizes (3.33 in / 7.0 in), fonts (Libertinus = ACM Biolinum look), colour roles
├── figures.py            fig1 … fig10   →  out/figs/*.pdf|png
├── tables.py             7 LaTeX tables →  out/tables/*.tex
├── arch/gemma_arch.tex   architecture (TikZ, built at final size)  →  arch/gemma_arch.pdf
├── preview/preview.tex   all figures + tables + draft captions at sigconf geometry → preview/preview.pdf
├── fonts/                Libertinus (SIL OFL); .ttf copies are for matplotlib, .otf for xelatex
└── logs_snapshot/        the logs these outputs were built from (offline reproduction)
```

## Rebuild on the server

```bash
cd /scratch/umni5/a/shives/POI_RecSys/GEMMA && conda activate gemma
bash paper/run_all.sh                       # ~1 min, CPU only
RAW=1 bash paper/run_all.sh                 # + raw-data analysis figures (A1–A8) in the same style, ~30 min
```

Offline (this snapshot): `python paper/collect.py --logs paper/logs_snapshot --analysis /none && python paper/figures.py && python paper/tables.py`.

## New runs that strengthen the paper (≈2.5 h on one GPU, all zero-shot, nothing retrained)

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_paper_ablate.sh > paper_ablate.out 2>&1 &
# then: bash paper/run_all.sh   → fig9_evidence_ablation, fig10_history and extra rows in tab_ablation appear
```

| Stage | What it answers | Output |
|---|---|---|
| E | which evidence carries the transfer (leave-one-out over frequency, recency, time, transition, trip, distance, pointer, activity path, anchors, *all evidence*) | `fig9`, `tab_ablation` |
| G | is the EPR gate needed (drop `g_S log S`, drop all personal gate terms, one global prior) — new test-time flag `--gate_mode` | `fig9`, `tab_ablation` |
| H | cold start: accuracy when only the k most recent check-ins are visible (k = 1…50) | `fig10_history` |

`--gate_mode` is new code in `gemma/hrem.py` (test-time only; default `""` = unchanged model). It has not been run on
a GPU here — check the first log for errors before leaving it overnight.

## Figure plan (8 pages of WWW long paper)

| # | File | Where | Message |
|---|---|---|---|
| 1 | `fig1_teaser` | figure*, p.1–2 | place IDs never transfer; the decision law does; one model ≈ in-city SOTA on 8 unseen cities and 1.9× LLM agents |
| 2 | `arch/gemma_arch` | figure*, §3 | identifier-free tokens → encoder → personal evidence → exact 3-decision mixture → train once, transfer |
| 3 | `fig3_main_forest` | figure*, §5.2 | per-city Δ vs STHGCN with CI: unseen (ZS + FT) and source |
| 4 | `fig4_llm` | column, §5.3 | GEMMA vs best LLM agent in each of the 8 unseen cities (×1.4–×2.7) |
| 5 | `fig5_anatomy` | figure*, §6 | returns make the accuracy; exploration is the open problem; where the signal is |
| 6 | `fig6_weights` | figure*, §6 | learned weights are interpretable, stable, gate ≈ −γ |
| 7 | `fig7_ablation` | column, §5.4 | learned θ +2.5 pts; backbone objectives ≈ 0 |
| 8 | `fig8_finetune` | appendix | fine-tuning gains are small and early |
| 9, 10 | `fig9_evidence_ablation`, `fig10_history` | §5.4 / appendix | after `run_paper_ablate.sh` |

Tables: `tab_datasets` (§4), `tab_main_unseen` + `tab_main_source` (§5.2, table*), `tab_llm` (§5.3, table*),
`tab_fairness` (§5.2 or appendix), `tab_ablation` (§5.4), `tab_weights` (appendix).
Paper preamble needs: `booktabs, multirow, xcolor[table]`, and `\definecolor{gemmarow}{HTML}{FDF0EA}`.

## Numbers the text may quote (all computed by `data_api.py`; Beijing excluded everywhere)

Unseen cities = Tokyo, Shanghai, Palembang, Melbourne, Sydney, Tangerang, Moscow, Bandung (25,144 test queries).

| | Acc@1 macro | Acc@1 pooled | NDCG@5 macro |
|---|---|---|---|
| GETNext (in-city) | .1640 | .1771 | .2431 |
| STHGCN (in-city) | .2274 | .2278 | .3149 |
| GEMMA zero-shot | .2250 | .2252 | **.3200** |
| GEMMA fine-tuned (train only) | **.2324** | **.2316** | **.3330** |
| GEMMA fine-tuned (train+val) | .2348 | .2334 | .3359 |
| LLM protocol: best LLM agent / GEMMA ZS | .2275 / .4338 (×1.91; ×1.4–×2.7 per city) | | |
| Source cities: STHGCN / GEMMA FT | .2055 / .1963 | .2073 / .1930 | .282 / .290 |

* Zero-shot GEMMA beats every supervised baseline except STHGCN in 8/8 unseen cities, is within 0.3 points of STHGCN
  on Acc@1 and above it on NDCG@5 (5/8 cities).
* Fine-tuned GEMMA is above STHGCN on every aggregate (+0.5 macro, +0.4 pooled, +1.8 NDCG@5; NDCG@5 ahead in 7/8).
* Per-city 95% CI vs STHGCN: zero-shot 1 win (Tangerang), 1 loss (Sydney), 6 ties; fine-tuned 1 win (Tangerang),
  7 ties, no loss; source cities 2 wins (Jakarta, New York), 3 losses (Istanbul, Petaling Jaya, Kuwait City), 1 tie.
* Learned vs hand-set evidence weights: +2.5 points (single seed). Fine-tuning moves the return weights by a median 8%.

Beijing is dropped in one place, `style.EXCLUDED`; `data_api.py` filters it from every run, report and analysis table,
and `analysis/common.py` drops it from the raw-data analyses (`GEMMA_WITH_BEIJING=1` restores it there).

## Caveats to state in the paper
* `fig1a` uses the A2 table from the old analysis run, whose cross-city averages still include Beijing pairs; the place-ID
  row is 1 bit by definition (city-local ids → disjoint supports). Re-run A2 (`RAW=1`, now without Beijing) before submission.
* `fig5c` uses the per-city A4 log lines (Acc@1 only) so Beijing can be dropped; Acc@5 returns after `RAW=1`.
* `fig7b` / backbone rows of `tab_ablation` come from the world-model ablation family (`final_R_ablation`), not v6.
* The hand-set vs learned comparison (`final_R_zero`) is a single seed.
* The LLM-protocol numbers for GEMMA use our own 5 × 200-user samples (the published LLM rows used one 200-user sample).
* Baseline numbers are parsed from the Massive-STEPS PDF text layer (v4 of the arXiv paper); spot-check them.
