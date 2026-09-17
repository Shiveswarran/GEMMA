"""gemma.config -- one dataclass of knobs; every experiment is a config diff."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    # ---- data ----
    data_root: str = "data"
    fsq_csv: str = "personalization-apis-movement-sdk-categories.csv"
    pretrain_cities: List[str] = field(default_factory=lambda: [
        # Massive-STEPS cities with > ~89k training trajectories
        "istanbul", "petaling_jaya", "jakarta", "new_york", "kuwait_city", "sao_paulo"])
    eval_cities: List[str] = field(default_factory=lambda: [
        "tokyo", "istanbul", "jakarta"])          # override on the CLI
    train_splits: List[str] = field(default_factory=lambda: ["train"])
    min_trail_len: int = 2
    min_user_trails: int = 3
    max_diary_len: int = 128                       # per-user diary cap (memory/speed)

    # ---- POI knowledge encoder (pluggable) ----
    poi_encoder: str = "hash"                      # hash | sbert | llm_cached
    poi_emb_dim: int = 384
    poi_emb_cache: str = "poi_emb_cache"           # dir for llm_cached / sbert cache
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- spatio-temporal encoding ----
    n_time_harmonics: int = 6                       # multi-scale Fourier tod/dow/season
    use_relative_space: bool = True                # log-dist + bearing (city-agnostic)

    # ---- backbone (JEPA) ----
    d_model: int = 256
    n_ctx_layers: int = 4
    n_pred_layers: int = 2
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.996
    geometry: str = "euclidean"                    # euclidean | poincare (ablation)
    # ---- two-tower next-POI ranker ----
    d_tower: int = 256
    nce_temp: float = 0.1
    n_hard_neg: int = 8                             # same-secondary hard negatives
    lambda_nce: float = 1.0
    # inference score blend (next-POI): tower.sim + hierarchy path + geo + temporal + revisit
    w_tower: float = 1.0
    w_hier: float = 1.0
    w_geo: float = 1.0
    w_temporal: float = 0.5
    w_revisit: float = 1.0
    hier_restrict_topk: int = 3                     # keep candidates whose activity in predicted top-k
    # ---- few-shot ----
    few_shot_trails_per_user: int = 2
    few_shot_epochs: int = 3

    # ---- SSL masking (ablation axis) ----
    masking: str = "block_future"                  # block_future | random_span | activity_block | causal_next
    mask_ratio: float = 0.25
    mask_block_max: int = 8

    # ---- objective weights ----
    lambda_pred: float = 1.0                        # JEPA latent MSE
    lambda_reg: float = 1.0                         # anti-collapse regularizer
    regularizer: str = "vicreg"                     # vicreg | sigreg | none
    vic_var: float = 1.0
    vic_cov: float = 0.04
    sig_n_proj: int = 128
    lambda_act: float = 1.0                         # next-activity CE (transferable)
    lambda_pri: float = 0.5                         # next-primary CE
    lambda_sec: float = 0.5                         # next-secondary CE
    lambda_motif: float = 0.3                       # motif codebook prior
    n_motifs: int = 24                              # motif codebook size (>=17)
    motif_dummy_nodes: bool = True                  # inject HOME/WORK dummy nodes

    # ---- optim ----
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 8
    seed: int = 42
    device: str = "auto"

    # ---- io ----
    out_dir: str = "runs"
    run_name: str = "gemma2"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))
