#!/usr/bin/env python3
"""Build deterministic, monthly blocked 3-fold BJT-date OOF manifest."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
seed = 20261015
z = np.load(ROOT / "clp_trainval.npz", allow_pickle=False)
idx = z["sample_indices"].astype(np.int64)
split = z["split"].astype(str)
station = z["station"].astype(str)
utc = z["target_time_utc"].astype(str)
clp = z["clp"][:, 0]
mask = z["common_mask"][:, 0].astype(bool)
train_ix = np.flatnonzero(split == "train")
assert len(train_ix) == 15414
assert np.all(idx[train_ix] < len(idx))
assert np.isin(clp[train_ix][mask[train_ix]], (0, 1, 2)).all()

bjt_date = {}
for i in train_ix:
    t = datetime.strptime(utc[i], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
    bjt_date[i] = t.strftime("%Y-%m-%d")

dates_by_month = defaultdict(set)
for i, d in bjt_date.items():
    dates_by_month[d[:7]].add(d)
date_fold = {}
for month in sorted(dates_by_month):
    dates = sorted(dates_by_month[month])
    # Keep each month's held-out dates contiguous. Randomly interleaving dates
    # makes the one-day leakage buffer remove nearly every neighboring fold.
    segments = np.array_split(np.asarray(dates, dtype=object), 3)
    # Rotate fold labels by month so short final segments do not always map to
    # the same fold; each fold still receives one contiguous segment per month.
    start = (int(month[-2:]) - 1) % 3
    for segment_id, segment in enumerate(segments):
        fold = (segment_id + start) % 3
        for d in segment.tolist():
            date_fold[d] = fold

fold_rows = []
for i in train_ix:
    d = bjt_date[i]
    fold_rows.append({
        "sample_index": int(idx[i]), "station": station[i], "target_time_utc": utc[i],
        "target_date_bjt": d, "month_bjt": d[:7], "fold": date_fold[d],
        "valid_pixels": int(mask[i].sum()),
        "clp0": int(np.count_nonzero(mask[i] & (clp[i] == 0))),
        "clp1": int(np.count_nonzero(mask[i] & (clp[i] == 1))),
        "clp2": int(np.count_nonzero(mask[i] & (clp[i] == 2))),
    })

# No BJT date or station-time key may cross folds.
folds_by_date = defaultdict(set)
folds_by_key = defaultdict(set)
for r in fold_rows:
    folds_by_date[r["target_date_bjt"]].add(r["fold"])
    folds_by_key[(r["station"], r["target_time_utc"])].add(r["fold"])
assert all(len(v) == 1 for v in folds_by_date.values())
assert all(len(v) == 1 for v in folds_by_key.values())
assert len({r["sample_index"] for r in fold_rows}) == len(fold_rows)
assert len({(r["station"], r["target_time_utc"]) for r in fold_rows}) == len(fold_rows)
summary = {"seed": seed, "folds": 3,
          "grouping": "contiguous BJT-date blocks within each month; both stations share a date fold",
          "train_samples": len(fold_rows), "train_dates": len(date_fold), "test_payloads_read": 0,
          "clp_sidecar_sha256": hashlib.sha256((ROOT / "clp_trainval.npz").read_bytes()).hexdigest(),
          "by_fold": {}}
for f in range(3):
    rs = [r for r in fold_rows if r["fold"] == f]
    summary["by_fold"][str(f)] = {
        "samples": len(rs), "dates": len({r["target_date_bjt"] for r in rs}),
        "stations": {s: sum(r["station"] == s for r in rs) for s in ("sili", "zhujia")},
        "valid_pixels": sum(r["valid_pixels"] for r in rs),
        "clp_counts": {str(c): sum(r[f"clp{c}"] for r in rs) for c in (0, 1, 2)},
    }
with (ROOT / "oof_train_folds_20261015_v2.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
    w.writeheader(); w.writerows(fold_rows)
(ROOT / "oof_train_folds_20261015_v2.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
