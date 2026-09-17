"""
gemma.fsq
---------
Parse the Foursquare category taxonomy (the CSV you exported:
columns `Category ID, Category Name, Category Label`, where Category Label is a
'>'-separated path, e.g. "Dining and Drinking > Restaurant > Indian Restaurant").

Level convention (as specified):
    activity   = top level of the path          (~10 classes, transferable)
    secondary  = leaf / Category Name           (finest category)
    primary    = the middle levels joined        (activity if no middle level)

This replaces the SBERT clustering of earlier GEMMA versions with the *real*,
named, reproducible hierarchy.  The Massive-STEPS `venue_category_id` column is
the Foursquare Category ID and is the join key.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd


def _levels_from_label(label: str):
    parts = [p.strip() for p in str(label).split(">")]
    parts = [p for p in parts if p]
    if not parts:
        return "Unknown", "Unknown", "Unknown"
    activity = parts[0]
    secondary = parts[-1]
    if len(parts) == 1:
        primary = parts[0]
    elif len(parts) == 2:
        primary = parts[0]              # top > leaf, no middle -> primary := activity
    else:
        primary = " > ".join(parts[1:-1])
    return activity, primary, secondary


class FSQTaxonomy:
    """Maps a Foursquare Category ID (string) -> (activity, primary, secondary)
    and to integer level ids with reserved 0 = <unk>."""

    def __init__(self, csv_path: Optional[str] = None):
        self.id2levels: Dict[str, tuple] = {}
        self.act2id = {"<unk>": 0}
        self.pri2id = {"<unk>": 0}
        self.sec2id = {"<unk>": 0}
        if csv_path and os.path.exists(csv_path):
            self.load(csv_path)

    def load(self, csv_path: str) -> "FSQTaxonomy":
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        id_col = "category_id"
        label_col = "category_label" if "category_label" in df.columns else df.columns[-1]
        for _, r in df.iterrows():
            cid = str(r[id_col]).strip()
            a, p, s = _levels_from_label(r[label_col])
            self.id2levels[cid] = (a, p, s)
            for name, table in ((a, self.act2id), (p, self.pri2id), (s, self.sec2id)):
                if name not in table:
                    table[name] = len(table)
        return self

    # -- lookups (fall back to <unk>=0 for ids not in the taxonomy) ---------
    def activity_id(self, category_id) -> int:
        lv = self.id2levels.get(str(category_id).strip())
        return self.act2id.get(lv[0], 0) if lv else 0

    def primary_id(self, category_id) -> int:
        lv = self.id2levels.get(str(category_id).strip())
        return self.pri2id.get(lv[1], 0) if lv else 0

    def secondary_id(self, category_id) -> int:
        lv = self.id2levels.get(str(category_id).strip())
        return self.sec2id.get(lv[2], 0) if lv else 0

    @property
    def n_activity(self) -> int:
        return len(self.act2id)

    @property
    def n_primary(self) -> int:
        return len(self.pri2id)

    @property
    def n_secondary(self) -> int:
        return len(self.sec2id)

    def coverage(self, category_ids) -> float:
        ids = [str(c).strip() for c in category_ids]
        hit = sum(1 for c in ids if c in self.id2levels)
        return hit / max(1, len(ids))
