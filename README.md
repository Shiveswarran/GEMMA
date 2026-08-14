# GEMMA — Global Embeddings for Mobility Modeling and Adaptation

A self-supervised, JEPA-style representation learner for **cross-city next-POI
recommendation**, implemented to plug into the
[Massive-STEPS](https://github.com/cruiseresearchgroup/Massive-STEPS) data and
evaluation protocol.

The model pretrains on a few "source" cities and transfers — **zero-shot** and
**few-shot**, with no per-city gradient updates — to unseen "target" cities by
retrieval in a shared activity space.

---

## What this implements (mapping to the paper)

| Paper concept | Where |
|---|---|
| Activity-space factorization `P_c(p|X) = P(a|A) · R_c(p|a)` | invariant encoder (grammar) + per-city POI table (realization) |
| City-agnostic check-in token (category, POI-name semantics, cyclic time, behavioural role) | `features.py`, `model.CheckinEmbed` |
| JEPA: predict masked-future **latents**, EMA target encoder | `model.GEMMA.forward` |
| SIGReg isotropic-Gaussian regularizer (anti-collapse) | `model.sigreg_loss` |
| Transfer by retrieval, not fine-tuning | `transfer.py` |
| Massive-STEPS eval (last check-in = target; Acc@{1,5,10}, NDCG@{5,10}) | `transfer.compute_metrics` |

### The added activity-transition mechanism (your request)

The paper *names* activities but doesn't give a mechanism to **learn activity
transitions**. This implementation adds one, in three grounded pieces:

1. **Activity-prototype codebook** (`ActivityCodebook`): `K` learnable prototypes.
   Each check-in latent is softly assigned to prototypes (SwAV-style, with
   Sinkhorn balancing to prevent prototype collapse). This is the learned,
   city-invariant "what activity is this" label.
2. **Transferable category grammar** (`cat_head`): an explicit head that predicts
   the **next check-in's shared primary category** from the JEPA-predicted latent.
   Because the category taxonomy is shared across cities, this is the component
   that actually transfers, and its accuracy — `cat_top1` — is the headline
   *grammar-transfer* diagnostic (grammar quality separated from realization).
3. **Interpretable transition operator `T`** (K×K, row-stochastic): a first-order
   activity grammar `P(a_i | a_{i-1})` ("coffee → office"), softly tied to the
   predictions via a Markov-consistency term. Saved as `transition_matrix.npy`;
   inspect it with `analyze_activities.py`.

Prototypes end up **category-grounded but finer** (e.g. splitting a category by
time-of-day / role), giving an interpretable activity vocabulary on top of the
raw taxonomy.

### Handling short trails

Massive-STEPS trails are short (median ~2–3 check-ins), which is too little for a
JEPA SSL signal per trail. We instead build a **per-user diary**: all of a user's
trails, time-ordered, concatenated with a learned `[SEP]` token between trails.
JEPA predicts masked *future* check-ins across this longer diary, so the user
representation and the activity grammar get a real training signal. Evaluation
still scores one trail's last check-in, exactly as the benchmark specifies.

---

## Install

```bash
pip install -r requirements.txt
```

`sentence-transformers` provides the frozen POI-name embeddings. If it is not
available, pass `--name_embedder hash` to use a deterministic offline fallback
(fine for pipeline testing; use SBERT for real results).

---

## Data layout

Point `--data_root` at a folder with one sub-folder per city, each containing the
three Massive-STEPS split CSVs you already have:

```
data/
  new_york/
    new_york_checkins_train.csv
    new_york_checkins_validation.csv
    new_york_checkins_test.csv
  tokyo/
    tokyo_checkins_train.csv
    ...
```

Expected columns (as in your dump): `trail_id, user_id, venue_id, latitude,
longitude, name, address, venue_category, venue_category_id,
venue_category_id_code, venue_city, venue_city_latitude, venue_city_longitude,
venue_country, timestamp`.

---

## Quick start

**0. Smoke test with no real data** (generates synthetic Massive-STEPS-format
cities with a *shared* grammar but disjoint POIs):

```bash
python make_synthetic_data.py --out data_synth
python run_experiment.py --data_root data_synth \
    --pretrain synth_a --transfer synth_b synth_c \
    --name_embedder hash --epochs 20 --run_name smoke
```

**1. The "glimpse": train on one city, test on two/three** (your first ask):

```bash
python run_experiment.py \
    --pretrain new_york \
    --transfer tokyo istanbul \
    --modes zero_shot few_shot \
    --epochs 20 --run_name glimpse
```

**2. Full paper setup** (pretrain jointly on the five >90k-trail cities, transfer
to the other ten):

```bash
python run_experiment.py \
    --pretrain istanbul petaling_jaya jakarta new_york kuwait_city \
    --transfer bandung beijing melbourne moscow palembang sao_paulo \
               shanghai sydney tangerang tokyo \
    --modes zero_shot few_shot \
    --epochs 40 --d_model 256 --n_activities 128 --run_name full
```

**3. Inspect the learned activity grammar:**

```bash
python analyze_activities.py --run_dir outputs/glimpse --city new_york
```

Every knob (loss weights, masking block size, kNN `k`, blend weights, model size,
priors) lives in `gemma/config.py` and is overridable from `run_experiment.py`.

---

## Outputs (per run, under `outputs/<run_name>/`)

| File | Contents |
|---|---|
| `config.json` | the exact config used |
| `pretrain.log`, `transfer.log` | full logs (per-step losses, per-city metrics) |
| `history.jsonl` | every training step's loss components + `cat_top1` |
| `gemma.pt` | model weights + vocab |
| `transition_matrix.npy` | K×K learned activity transition operator |
| `metrics_summary.json` | Acc@{1,5,10}, NDCG@{5,10}, MRR per city × mode |
| `predictions_<city>_<mode>.csv` | **per test trail: the top-10 ranked POIs**, their scores, `rank_of_truth`, `hit@{1,5,10}`, and predicted next category |

---

## Evaluation protocol (matches the repo)

* One **trail** = check-ins sharing a `trail_id`, time-ordered; the **last**
  check-in is the target, the rest are the context prefix.
* Predictions are ranked over the **target city's POI vocabulary**.
* `NDCG@k = 1/log2(rank+2)` if the truth is in the top-k (the repo's convention),
  averaged over trails; `MRR = 1/(rank+1)`.
* **zero-shot**: retrieval pool = source cities only; no target labels, no
  popularity prior. **few-shot**: a few labelled target trails per user are added
  to the retrieval pool and a city popularity prior is switched on — still no
  gradient updates.

---

## Honest notes / where to push next

* **Absolute magnitudes.** Zero-shot next-POI over a full city vocabulary
  (thousands of venues, no target supervision) is genuinely hard; expect modest
  Acc@1. The meaningful signals are (a) `cat_top1` (does the activity grammar
  transfer?) and (b) few-shot > zero-shot, and wins in **cold-start / low-
  regularity** regimes where ID-based models have nothing to memorize. Segmenting
  the prediction CSVs by seen-vs-new venue and by user predictability is the
  analysis that makes GEMMA's case.
* **Spatial features.** By design the invariant encoder uses *relative* motion
  (distance/bearing to the previous check-in), not absolute coordinates, to keep
  it transferable (`use_absolute_coords_in_encoder=False`). City/country live in
  the realization side. Flip the flag to ablate.
* **The transition operator `T`** only sharpens once predictions are confident;
  raise `lambda_markov` and train longer if you want a crisper interpretable `T`.
* **Baselines.** For a fair table against GETNext / STHGCN / KAS / CATUS, run
  those through this same split + metric code (the repo already hosts several).

---

## Module map

```
gemma/
  config.py     one dataclass of all settings (serialized with every run)
  utils.py      logging, seeding, haversine/bearing
  features.py   SBERT name/category embeddings (+ offline fallback),
                cyclic temporal encoding, relative spatial features, category vocab
  data.py       Massive-STEPS loader, per-user diaries with [SEP], feature
                precompute, JEPA windowing dataset, per-trail eval structures
  model.py      GEMMA: check-in embed + role router, context/EMA-target
                Transformers, predictor, activity codebook + T, cat grammar, SIGReg
  pretrain.py   joint JEPA + category-grammar + SIGReg training loop
  transfer.py   retrieval pools, candidate POI embedding, zero/few-shot ranking,
                metrics, per-trail top-K saving
run_experiment.py    end-to-end pretrain -> transfer -> evaluate
analyze_activities.py interpret T and prototype->category composition
make_synthetic_data.py  tiny Massive-STEPS-format data for offline testing
```
