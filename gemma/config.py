"""All hyper-parameters of GEMMA in one dataclass.

A run directory stores its configuration as ``config.json``; every script loads it from there
and command-line flags override individual fields.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List

SOURCE_CITIES = ["istanbul", "petaling_jaya", "jakarta", "new_york", "kuwait_city", "sao_paulo"]
TARGET_CITIES = ["tokyo", "shanghai", "palembang", "melbourne", "sydney", "tangerang", "moscow",
                 "bandung"]


@dataclass
class Config:
    # ---------------------------------------------------------------- data
    data_root: str = "data"
    fsq_csv: str = "personalization-apis-movement-sdk-categories.csv"   # Foursquare taxonomy
    pretrain_cities: List[str] = field(default_factory=lambda: list(SOURCE_CITIES))
    train_splits: List[str] = field(default_factory=lambda: ["train"])
    min_trail_len: int = 2
    min_user_trails: int = 3              # pretraining only
    max_diary_len: int = 128              # pretraining only

    # ---------------------------------------------------------------- place text
    poi_encoder: str = "cached"           # cached (precomputed sentence embeddings) | hash (tests)
    poi_emb_dim: int = 384
    poi_emb_cache: str = "poi_emb_cache"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------------------------------------------------------- check-in tokens
    n_time_harmonics: int = 6             # Fourier features of hour, weekday and month

    # ---------------------------------------------------------------- encoder
    d_model: int = 256
    n_ctx_layers: int = 4
    n_pred_layers: int = 2
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.996

    # ---------------------------------------------------------------- semantic IDs
    use_semantic_id: bool = True
    sid_levels: int = 4                   # 3 residual-quantization levels + 1 disambiguation level
    sid_codebook: int = 256
    sid_cache: str = "sid_cache"

    # ---------------------------------------------------------------- self-supervised pretraining
    masking: str = "mixed"                # mixed | block_future | trail_end
    mask_ratio: float = 0.25
    mask_block_max: int = 8
    regularizer: str = "vicreg"           # vicreg | none
    vic_var: float = 1.0
    vic_cov: float = 0.04
    lambda_reg: float = 1.0
    lambda_motif: float = 0.3
    n_motifs: int = 24
    motif_dummy_nodes: bool = True
    uncertainty_weighting: bool = True
    uw_min_w: float = 0.5
    lambda_sid: float = 2.0
    lambda_pred: float = 1.0              # fixed task weights (used when uncertainty_weighting=False)
    lambda_act: float = 1.0
    lambda_pri: float = 0.5
    lambda_sec: float = 0.5
    lambda_geo: float = 1.0
    lambda_temporal: float = 1.0
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0

    # ---------------------------------------------------------------- next-place model
    d_real: int = 256                     # dimension of the query / place-content space
    hist_trails: int = 15                 # most recent history trajectories in the encoder input
    hist_max_tokens: int = 200            # token cap of [history SEP ... SEP prefix]
    hist_last: int = -1                   # keep only the user's last k visible trajectories (-1 = all)
    hist_budget: int = -1                 # causal history: only the k most recent trajectories that
                                          # start before the query (-1 = off, 0 = no history)
    eval_protocol: str = "sthgcn"         # sthgcn: score targets whose place and user occur in the
                                          # training split | all: every test trajectory
    eval_split: str = "test"
    full_visible: str = "train"           # fine-tuned setting, visible at test time: train | train+val

    # ---------------------------------------------------------------- test-time ablations
    drop_ret: str = ""                    # comma list of return evidence switched off
    drop_exp: str = ""                    # comma list of explore evidence switched off
    gate_mode: str = ""                   # '' | noS | nopersonal | const
    evidence: str = "full"                # full | none: no personal statistics at all; known places
                                          # are those in the encoder input, reached only through
                                          # the latent pointer and distance
    out_tag: str = ""                     # suffix of result files (ablations)

    # ---------------------------------------------------------------- training of the mixture
    ft_epochs: int = 8                    # fine-tuning on a target city (early stopping)
    ft_lr_head: float = 2e-3
    ft_lr_backbone: float = 1e-4
    ft_lr_resid: float = 3e-4             # per-place / per-user residual embeddings
    ft_batch: int = 128
    ft_max_queries: int = 400000          # queries per city and epoch (subsampled above this)
    ft_all_positions: bool = True         # train on every next-step position of a trajectory
    val_queries: int = 3000
    val_every: int = 500                  # validate every N steps (0 = end of epoch only)
    patience: int = 3
    log_every: int = 200

    # ---------------------------------------------------------------- system
    amp: bool = True
    num_workers: int = 8
    seed: int = 42
    device: str = "auto"
    allow_cpu: bool = False
    out_dir: str = "runs"
    run_name: str = "gemma"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            raw = json.load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
