

    python -m pytest tests -q

1. Exact normalization: sum_v P(v | xi) = 1 over all places of the city, for every query.
2. Identifier invariance: relabelling the places of a city (same metadata, new identifiers)
   leaves every probability unchanged.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gemma.backbone import GEMMAEncoder  # noqa: E402
from gemma.city import CityIndex  # noqa: E402
from gemma.config import Config  # noqa: E402
from gemma.data import build_city_data  # noqa: E402
from gemma.model import forward, new_head  # noqa: E402
from gemma.pipeline import split_ids  # noqa: E402
from gemma.taxonomy import Taxonomy  # noqa: E402
from gemma.text_encoder import TextEncoder  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_synthetic.py"), "--out", str(d),
                    "--fsq_out", str(d / "taxonomy.csv"), "--cities", "city", "--n_users", "20",
                    "--days", "8"], check=True, capture_output=True)
    # relabelled copy: same check-ins, every venue id replaced by a random new identifier
    rng = np.random.RandomState(0)
    os.makedirs(d / "relab", exist_ok=True)
    frames = {sp: pd.read_csv(d / "city" / f"city_checkins_{sp}.csv") for sp in ("train", "validation", "test")}
    ids = np.unique(np.concatenate([f["venue_id"].values for f in frames.values()]))
    new = dict(zip(ids, rng.permutation(10 ** 6)[:len(ids)] + 10 ** 6))
    for sp, f in frames.items():
        f = f.copy(); f["venue_id"] = f["venue_id"].map(new)
        f.to_csv(d / "relab" / f"relab_checkins_{sp}.csv", index=False)
    cfg = Config(data_root=str(d), fsq_csv=str(d / "taxonomy.csv"), poi_encoder="hash", poi_emb_dim=32,
                 d_model=32, n_ctx_layers=1, n_pred_layers=1, n_heads=2, use_semantic_id=False,
                 poi_emb_cache=str(d / "emb"), sid_cache=str(d / "sid"))
    tax, te = Taxonomy(cfg.fsq_csv), TextEncoder(cfg)
    torch.manual_seed(0)
    out = {}
    for name in ("city", "relab"):
        city = build_city_data(cfg, name, ["train", "validation", "test"], tax, te, filter_users=False)
        out[name] = city
    enc = GEMMAEncoder(cfg, out["city"].cont_dim, tax.n_activity, tax.n_primary, tax.n_secondary).eval()
    return cfg, enc, out, new


def _model(cfg, enc, city):
    ids = split_ids(city)
    ci = CityIndex(city, observed=ids["train"], stat=None, cfg=cfg, device="cpu")
    torch.manual_seed(1)
    head = new_head(enc, ci, cfg, "cpu").eval()
    with torch.no_grad():                     # non-trivial weights in every term
        for p in head.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return ci, head, ci.queries(ids["test"], True)[:40]


def test_normalization(setup):
    cfg, enc, cities, _ = setup
    ci, head, qs = _model(cfg, enc, cities["city"])
    with torch.no_grad():
        lp = forward(enc, head, ci, qs)["logits"]
    total = torch.logsumexp(lp, dim=1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-4), total


def test_identifier_invariance(setup):
    cfg, enc, cities, new = setup
    ci_a, head, qs = _model(cfg, enc, cities["city"])
    ci_b, _, qs_b = _model(cfg, enc, cities["relab"])
    head_b = new_head(enc, ci_b, cfg, "cpu").eval()
    head_b.load_state_dict(head.state_dict())
    assert qs == qs_b
    with torch.no_grad():
        lp_a = forward(enc, head, ci_a, qs)["logits"]
        lp_b = forward(enc, head_b, ci_b, qs)["logits"]
    # place k of city A is place perm[k] of the relabelled city
    vid_a = cities["city"].poi_table["venue_ids"]
    row_b = cities["relab"].poi_table["id2row"]
    perm = torch.as_tensor([row_b[int(new[int(v)])] for v in vid_a])
    assert torch.allclose(lp_a, lp_b[:, perm], atol=1e-4)
