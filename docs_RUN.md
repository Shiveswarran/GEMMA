# GEMMA v2 — run procedure (semantic-ID + generative-diffusion leaf)

## 0. Environment  (keep all caches OFF your home quota)
```bash
conda activate gemma
export HF_HOME=/scratch/umni5/a/$USER/hf; export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
pip install -r requirements.txt
```

## 1. Data  (Massive-STEPS splits + FSQ taxonomy CSV)
CSV="personalization-apis-movement-sdk-categories.csv"   # your real FSQ file
Offline smoke: `python scripts/make_synthetic.py --out data --cities src_a src_b tgt_c`
(writes a tiny taxonomy CSV of the same name)

## 2. Precompute LLM POI embeddings  (poi_encoder=llm_cached)
```bash
python scripts/embed_pois.py --data_root data \
  --cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
           tokyo bandung beijing melbourne moscow palembang shanghai sydney tangerang \
  --cache poi_emb_cache --model BAAI/bge-small-en-v1.5
```

## 3. Build semantic IDs  (RQ-VAE, ONCE; frozen codebooks -> transferable)
```bash
python scripts/build_semantic_ids.py --data_root data --fsq_csv "$CSV" \
  --train_cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
  --all_cities   istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
                 tokyo bandung beijing melbourne moscow palembang shanghai sydney tangerang \
  --poi_encoder llm_cached --levels 3 --codebook 256 --epochs 30 --cache sid_cache
```
Check the log: "distinct semantic IDs" high, "collision rate" low (few %).

## 4. Pretrain the backbone  (diffusion leaf + multi-feature JEPA, ONCE)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py --data_root data --fsq_csv "$CSV" \
  --pretrain_cities istanbul petaling_jaya jakarta new_york kuwait_city sao_paulo \
  --poi_encoder llm_cached --leaf_head diffusion --masking block_future \
  --regularizer vicreg --geometry euclidean \
  --epochs 40 --batch_size 256 --d_model 256 --num_workers 8 --run_name gemma_v2
```
Watch: sid/geo/tmp losses falling; act@1 >= pri@1 >= sec@1; w_* (learned task weights) in history.jsonl.

## 5. Evaluate  (2 tasks x 3 shots x 15 cities)
```bash
python scripts/run_downstream.py --run_dir runs/gemma_v2 --data_root data \
  --cities tokyo istanbul jakarta new_york kuwait_city petaling_jaya sao_paulo \
           bandung beijing melbourne moscow palembang shanghai sydney tangerang \
  --tasks next_poi next_activity --shots zero_shot few_shot full --device auto
```
-> runs/gemma_v2/benchmark_summary.json + printed table.

## 6. Ablations (the leaf-head + design grid; one flag per run, unique --run_name)
```bash
# leaf head: generative-diffusion vs autoregressive vs retrieval two-tower
python scripts/run_pretrain.py ... --leaf_head ar        --run_name abl_leaf_ar
python scripts/run_pretrain.py ... --leaf_head retrieval --no_semantic_id --run_name abl_leaf_retrieval
# no semantic id / no uncertainty weighting / masking / regularizer / geometry / poi encoder
python scripts/run_pretrain.py ... --no_semantic_id      --run_name abl_no_sid
python scripts/run_pretrain.py ... --no_uncertainty      --run_name abl_no_uw
python scripts/run_pretrain.py ... --masking random_span --run_name abl_mask_random
python scripts/run_pretrain.py ... --regularizer sigreg  --run_name abl_reg_sigreg
python scripts/run_pretrain.py ... --geometry poincare   --run_name abl_geo_poincare
python scripts/run_pretrain.py ... --sid_codebook 512    --run_name abl_sid_k512   # via config
```
Then run step 5 per run_dir. Full grid: experiments/matrix.md.
