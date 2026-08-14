"""
gemma.config
------------
A single dataclass that holds every knob for the GEMMA pipeline (data,
features, model, JEPA/activity objective, optimisation, transfer/eval).

Keeping everything in one typed object makes runs reproducible: the config
is serialised to JSON next to every checkpoint and every results file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class GemmaConfig:
    # ---------------------------------------------------------------- data
    data_root: str = "data"                 # folder with one sub-folder per city
    # city sub-folder names must match the file prefixes, e.g.
    #   data/new_york/new_york_checkins_train.csv
    pretrain_cities: List[str] = field(default_factory=lambda: ["new_york"])
    transfer_cities: List[str] = field(default_factory=lambda: ["tokyo"])

    min_trail_len: int = 2                  # drop trails with < this many check-ins
    min_user_trails: int = 3                # drop users with < this many trails
    max_user_seq_len: int = 128             # cap the per-user concatenated sequence
    max_context_len: int = 64               # context window fed to the encoder

    # ------------------------------------------------------------ features
    name_embedder: str = "sbert"            # "sbert" | "hash" (hash = offline fallback)
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    name_emb_dim: int = 384                 # 384 for MiniLM; auto-set for hash
    cat_sbert: bool = True                  # also embed the category string with SBERT
    coarse_activities: bool = True          # induce a COARSE shared activity label
    n_coarse_activities: int = 12           # #coarse activity classes (grammar target)
    n_time_harmonics: int = 4               # Fourier harmonics per cyclic time field
    use_absolute_coords_in_encoder: bool = False  # keep False for city-invariance
    coord_grid_m: float = 500.0             # metres, for optional coarse geo-hashing

    # ------------------------------------------------------------- model
    d_model: int = 128
    n_ctx_layers: int = 3
    n_pred_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    n_roles: int = 8                        # behavioural-role router codes
    n_activities: int = 64                  # activity-prototype codebook size (K)
    activity_temp: float = 0.5              # softmax temperature for prototype assign

    # -------------------------------------------------- JEPA / objective
    ema_decay: float = 0.996
    mask_block_min: int = 1                 # masked future block size (check-ins)
    mask_block_max: int = 4
    lambda_pred: float = 1.0                # JEPA latent-prediction loss
    lambda_cat: float = 1.0                 # next-CATEGORY grammar (transferable)
    lambda_act: float = 0.3                 # latent activity-prototype clustering
    lambda_markov: float = 0.1              # tie grammar to Markov operator T
    lambda_sig: float = 0.05                # SIGReg isotropic-Gaussian regulariser
    lambda_cb: float = 0.05                 # codebook usage entropy regulariser
    sig_n_proj: int = 128                   # random projections for SIGReg
    sigreg_warmup_epochs: int = 1           # warm-start: predictor first, reg later

    # ------------------------------------------------------- optimisation
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_steps: int = 200
    seed: int = 42
    device: str = "auto"                    # "auto" | "cpu" | "cuda"
    num_workers: int = 0
    amp: bool = True                        # mixed precision on CUDA (bf16/fp16)
    compile: bool = False                   # torch.compile the model (opt-in)
    pin_memory: bool = True                 # pinned host memory for H2D copies

    # ---------------------------------------------------------- transfer
    knn_k: int = 50                         # neighbours in source pool retrieval
    retrieval_blend: float = 0.5            # weight on retrieval vs grammar head
    sem_blend: float = 0.5                  # weight on semantic-sim vs activity-match
    personal_prior: float = 1.0            # bonus for POIs already in the query prefix
    geo_prior: float = 1.2                  # weight on geographic proximity to last check-in
    popularity_prior: float = 0.15          # weight on city popularity prior (few-shot)
    few_shot_trails_per_user: int = 1       # labelled target trails added per user
    top_k_save: int = 10                    # how many ranked POIs to persist per trail
    eval_ks: List[int] = field(default_factory=lambda: [1, 5, 10])
    ndcg_ks: List[int] = field(default_factory=lambda: [5, 10])
    max_eval_candidates: int = 0            # 0 = rank over full city vocab

    # ------------------------------------------------------------- io
    out_dir: str = "outputs"
    run_name: str = "gemma_run"
    log_every: int = 20
    save_predictions: bool = True

    # ---------------------------------------------------------- helpers
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "GemmaConfig":
        with open(path) as f:
            return cls(**json.load(f))
