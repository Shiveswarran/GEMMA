# GEMMA v2 — run procedure (step by step)

## 0. Environment
```bash
conda create -n gemma python=3.11 -y && conda activate gemma
pip install -r requirements.txt
# CUDA torch matching your driver, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 1. Data
- Massive-STEPS splits: data/<city>/<city>_checkins_{train,validation,test}.csv
- Foursquare taxonomy CSV (Category ID, Category Name, Category Label) at --fsq_csv
Offline smoke data + tiny taxonomy:
```bash
python scripts/make_synthetic.py --out data --cities src_a src_b tgt_c
```

## 2. (optional) Precompute modern-LLM POI embeddings  (poi_encoder=llm_cached)
```bash
python scripts/embed_pois.py --data_root data \
  --cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
           tokyo bandung beijing melbourne moscow palembang shanghai sydney tangerang \
  --cache poi_emb_cache --model <your-LLM-encoder>   # e.g. Alibaba-NLP/gte-Qwen2-1.5B-instruct
```

## 3. Pretrain the GEMMA backbone (ONCE, on the 6 high-resource cities)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --data_root data --fsq_csv personalization-apis-movement-sdk-categories.csv \
  --pretrain_cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
  --poi_encoder llm_cached --masking block_future --regularizer vicreg \
  --geometry euclidean --epochs 40 --batch_size 256 --d_model 256 \
  --num_workers 8 --run_name gemma_v2
```
Watch runs/gemma_v2/pretrain.log for: act@1 >= pri@1 >= sec@1 (transfer gradient),
nce decreasing, poi@1 rising.

## 4. Downstream — two tasks x three shots x all cities
```bash
python scripts/run_downstream.py --run_dir runs/gemma_v2 --data_root data \
  --cities tokyo istanbul jakarta new_york kuwait_city petaling_jaya sao_paulo \
           bandung beijing melbourne moscow palembang shanghai sydney tangerang \
  --tasks next_poi next_activity --shots zero_shot few_shot full --device auto
```
Outputs: runs/gemma_v2/down_<city>_<task>_<shot>.json  and  benchmark_summary.json,
with the printed BENCHMARK SUMMARY table (acc@1/@5, ndcg@5, mrr, per-level act/pri/sec).

## 5. Ablations (paper grid, see experiments/matrix.md)
Re-run step 3 with one axis changed, keeping --run_name unique, then step 4:
```bash
# masking
for M in block_future random_span activity_block causal_next; do
  python scripts/run_pretrain.py ... --masking $M --run_name abl_mask_$M ; done
# regularizer / geometry / poi_encoder likewise
python scripts/run_pretrain.py ... --regularizer sigreg  --run_name abl_reg_sigreg
python scripts/run_pretrain.py ... --geometry   poincare --run_name abl_geo_poincare
python scripts/run_pretrain.py ... --poi_encoder sbert   --run_name abl_poi_sbert
```

## 6. External SOTA adapters (next step, needs each repo's model file)
Export GEMMA POI features for a repo:
```python
from gemma.downstream import ExternalAdapter
feat = ExternalAdapter(backbone, cfg, device).export(city)   # {'poi_feat','poi_ids'}
```
Then apply the 3-line embedding-table patch in that repo (README 'Integrating a SOTA model').
Build order: GETNext -> STHGCN -> LSTPM -> DeepMove -> Graph-Flashback -> Flashback.
