# GEMMA: Global Embeddings for urban Mobility Modeling and Adaptation

Code for the paper *GEMMA: Global Embeddings for urban Mobility Modeling and Adaptation,* (under review).

GEMMA predicts a person's next place in a city it has never seen, without any training data
from that city. It does not learn place identifiers, which are different in every city.
Instead, it learns how people decide where to go next, which works the same way in every city.
The next-place distribution factorizes exactly into three decisions:

```
P(v | context) = P(activity a_v) · [ (1 − π_a) · P_return(v)   if v is a place the user knows
                                      π_a      · P_explore(v)  otherwise ]
```

- **Activity:** an encoder over identifier-free check-in tokens predicts the next activity
  (time, relative geometry, a three-level activity taxonomy and a text embedding of the place).
- **Return or explore:** a gate π_a decides this. Its parameterization contains the
  exploration–preferential-return law (π ∝ S^−γ) as a special case.
- **Which place:** within each channel, places are scored by context-modulated, non-negative
  weights on statistics of the user's own history: frequency, recency, hour and weekday, the
  current trip, personal transitions, distance and a latent pointer.

No parameter is indexed by a place. A model trained on six source cities therefore applies
unchanged to any other city.

## Results (Massive-STEPS, 8 unseen cities, 3 seeds)

| Model | trained on the target city | Acc@1 (macro) | Acc@1 (pooled) | NDCG@5 |
|---|---|---|---|---|
| STHGCN (best supervised baseline) | yes | 0.227 | 0.228 | 0.315 |
| GETNext | yes | 0.164 | – | – |
| **GEMMA, zero-shot** | **no** | 0.225 | 0.225 | **0.320** |
| **GEMMA, fine-tuned** | yes | **0.232** | **0.232** | **0.333** |

The LLM-Move protocol ranks the true place among 100 random places. On it, zero-shot GEMMA
reaches Acc@1 0.434. The best of 15 LLM-agent configurations, chosen separately in each city,
reaches 0.228.

## Installation

```bash
conda create -n gemma python=3.10 -y && conda activate gemma
pip install -r requirements.txt
python -m pytest tests -q        # exact normalization + identifier invariance (CPU, seconds)
```

## Data

1. **Massive-STEPS:** download the check-in splits from the
   [Massive-STEPS repository](https://github.com/cruiseresearchgroup/Massive-STEPS) and place
   them as `data/<city>/<city>_checkins_{train,validation,test}.csv`.
2. **Foursquare category taxonomy:** a CSV with the columns
   `Category ID, Category Name, Category Label`. The label is the `>`-separated category path.
   It is published by Foursquare with its Places API documentation. Pass it with `--fsq_csv`
   (default name: `personalization-apis-movement-sdk-categories.csv`).

Source cities: Istanbul, Petaling Jaya, Jakarta, New York, Kuwait City, São Paulo.
Unseen target cities: Tokyo, Shanghai, Palembang, Melbourne, Sydney, Tangerang, Moscow, Bandung.

## Quick start on synthetic data (CPU, about 3 minutes)

```bash
python scripts/make_synthetic.py --out data_synth --fsq_out data_synth/taxonomy.csv \
    --cities src_a src_b tgt_c
python scripts/build_semantic_ids.py --data_root data_synth --fsq_csv data_synth/taxonomy.csv \
    --train_cities src_a src_b --all_cities src_a src_b tgt_c --codebook 16 --epochs 3
python scripts/pretrain.py --data_root data_synth --fsq_csv data_synth/taxonomy.csv \
    --pretrain_cities src_a src_b --poi_encoder hash --epochs 2 --batch_size 32 --d_model 64 \
    --num_workers 0 --run_name synth --allow_cpu
python scripts/train_source.py --run_dir runs/synth --seed 0 --epochs 1 --batch 32 --allow_cpu
python scripts/evaluate.py --run_dir runs/synth --source runs/synth/source_s0.pt --cities tgt_c \
    --settings zero_shot finetuned --protocols full llm --ft_epochs 1 --allow_cpu
```

## Reproducing the paper

`scripts/reproduce.sh` runs every stage. Each stage can also be run alone, e.g.
`ONLY="eval report" bash scripts/reproduce.sh`.

| Stage | Command (see the script for all arguments) | Output |
|---|---|---|
| prep | `embed_pois.py`, `build_semantic_ids.py` | text embeddings of places, semantic IDs (3 RQ-VAE levels + 1 disambiguation level) |
| pretrain | `pretrain.py` | self-supervised encoder `runs/gemma/gemma.pt` |
| source | `train_source.py --seed S` | zero-shot model `runs/gemma/source_sS.pt` (source cities only) |
| eval | `evaluate.py --settings zero_shot finetuned` | Table 2, per-query ranks |
| llm | `evaluate.py --protocols llm --hist_last 15 --eval_protocol all` | LLM-agent comparison |
| ablate | `evaluate.py` with the switches below | evidence, gate and history ablations |
| report | `report.py`, `mobility_laws.py` | per-city tables with 95% intervals, γ per city vs −g_S |

**Settings.**
- **Zero-shot:** no update and no population statistics of the target city. Each test user's
  training-split trajectories are visible, as in the prompts of LLM agents.
- **Fine-tuned:** the zero-shot model is adapted on the target city's training split.
  Population statistics of that split are added, along with zero-initialized per-place and
  per-user residual embeddings. Only the training split is visible at test time
  (`--full_visible train`, the default).

**Protocol.** By default, test queries whose target place or user does not occur in the
training split are removed, as in the protocol of the supervised baselines
(`--eval_protocol sthgcn`). `--eval_protocol all` scores every test trajectory.

Massive-STEPS does not require training trajectories to precede test trajectories, so a
user's history can contain trajectories recorded after the query; this holds for every model
on the benchmark. `--hist_budget k` restricts the history to the k most recent trajectories
that start before the query, which gives a strictly causal evaluation.

**Test-time ablations** (`scripts/evaluate.py`, no retraining):

| Flag | Effect |
|---|---|
| `--drop_ret hist,rec,tod,dow,ptrans,prefix,is_last,geo,lptr` | switch off return evidence |
| `--drop_exp hier,geo,anchor` | switch off explore evidence |
| `--evidence none` | no personal statistics at all |
| `--gate_mode noS \| nopersonal \| const` | drop the exploration-law term / all personal gate terms / use one global prior |
| `--hist_budget k` | causal history of k trajectories |
| `--source none` | hand-set initial evidence weights instead of the source-trained ones |

`train_source.py --random_encoder` trains the mixture on an encoder without pretraining.

## Repository layout

```
gemma/
  config.py        all hyper-parameters
  taxonomy.py      Foursquare category path -> activity / primary / leaf
  text_encoder.py  cached sentence embeddings of place texts
  data.py          check-in features, trajectories, pretraining diaries
  semantic_id.py   RQ-VAE for place semantic IDs
  backbone.py      encoder + self-supervised (JEPA) pretraining objective
  pretrain.py      pretraining loop
  city.py          per-city index: histories, queries, personal and population evidence
  model.py         the exact activity / return-explore mixture and its loss
  train.py         source training and fine-tuning loop with early stopping
  evaluate.py      supervised and LLM-Move protocols, return/explore decomposition
  pipeline.py      loading checkpoints, zero-shot / fine-tuned evaluation of one city
scripts/           command-line entry points (see the table above)
tests/             properties of the mixture
```

## Hardware

All experiments run on one GPU with 24 GB of memory. Training on CPU is refused unless
`--allow_cpu` is given.

## License

MIT (see `LICENSE`). Massive-STEPS and the Foursquare taxonomy are subject to their own
licenses.
