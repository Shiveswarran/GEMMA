"""Three-level activity taxonomy from the Foursquare category file.

The file has the columns ``Category ID, Category Name, Category Label``; the label is a
``>``-separated path such as ``Dining and Drinking > Restaurant > Indian Restaurant``.

    activity  = first element of the path   (about ten classes, shared by all cities)
    primary   = the middle elements          (the activity if the path has no middle)
    leaf      = last element                 (the finest category)

Massive-STEPS stores the Foursquare category id in ``venue_category_id``; it is the join key.
Index 0 of every level is reserved for unknown categories.
"""
from __future__ import annotations

import os

import pandas as pd


def _levels(label: str):
    parts = [p.strip() for p in str(label).split(">") if p.strip()]
    if not parts:
        return "Unknown", "Unknown", "Unknown"
    activity, leaf = parts[0], parts[-1]
    primary = parts[0] if len(parts) <= 2 else " > ".join(parts[1:-1])
    return activity, primary, leaf


class Taxonomy:
    def __init__(self, csv_path: str | None = None):
        self.id2levels = {}
        self.act2id, self.pri2id, self.sec2id = {"<unk>": 0}, {"<unk>": 0}, {"<unk>": 0}
        if csv_path and os.path.exists(csv_path):
            self.load(csv_path)

    def load(self, csv_path: str) -> "Taxonomy":
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        label_col = "category_label" if "category_label" in df.columns else df.columns[-1]
        for _, r in df.iterrows():
            a, p, s = _levels(r[label_col])
            self.id2levels[str(r["category_id"]).strip()] = (a, p, s)
            for name, table in ((a, self.act2id), (p, self.pri2id), (s, self.sec2id)):
                table.setdefault(name, len(table))
        return self

    def _lookup(self, category_id, level, table):
        lv = self.id2levels.get(str(category_id).strip())
        return table.get(lv[level], 0) if lv else 0

    def activity_id(self, category_id) -> int:
        return self._lookup(category_id, 0, self.act2id)

    def primary_id(self, category_id) -> int:
        return self._lookup(category_id, 1, self.pri2id)

    def secondary_id(self, category_id) -> int:
        return self._lookup(category_id, 2, self.sec2id)

    @property
    def n_activity(self) -> int:
        return len(self.act2id)

    @property
    def n_primary(self) -> int:
        return len(self.pri2id)

    @property
    def n_secondary(self) -> int:
        return len(self.sec2id)
