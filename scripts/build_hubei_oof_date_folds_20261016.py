#!/usr/bin/env python3
"""Freeze train-only, month-stratified date folds for Hubei S->R OOF inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
FOLDS = 5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    if sha(a.split) != EXPECTED_SPLIT_SHA:
        raise ValueError("Frozen Hubei date assignment checksum mismatch")
    with a.split.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"date_BJT", "split"}.issubset(rows[0]):
        raise ValueError("Unexpected date-assignment schema")
    dates = [r["date_BJT"] for r in rows if r["split"] == "train"]
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate train dates in frozen split")
    by_month: dict[str, list[str]] = {}
    for date in dates:
        if len(date) != 10 or date[4] != "-" or date[7] != "-":
            raise ValueError(f"Invalid BJT date {date!r}")
        month = date[:7]
        by_month.setdefault(month, []).append(date)
    assignment: dict[str, int] = {}
    month_counts: dict[str, dict[str, int]] = {}
    global_counts = Counter()
    for month, month_dates in sorted(by_month.items()):
        ranked = sorted(month_dates, key=lambda d: hashlib.sha256(
            f"hubei-oof-datefold-v1|{month}|{d}".encode("ascii")).hexdigest())
        counts = Counter()
        tie_start = int(hashlib.sha256(f"hubei-oof-rotation-v1|{month}".encode("ascii")).hexdigest()[:8], 16) % FOLDS
        offsets = sorted(range(FOLDS), key=lambda x: ((x - tie_start) % FOLDS))
        offset = min(offsets, key=lambda off: sum(
            (global_counts[f] + sum((i + off) % FOLDS == f for i in range(len(ranked)))) ** 2
            for f in range(FOLDS)))
        for i, date in enumerate(ranked):
            fold = (i + offset) % FOLDS
            assignment[date] = fold
            counts[str(fold)] += 1
            global_counts[fold] += 1
        month_counts[month] = {str(f): int(counts[str(f)]) for f in range(FOLDS)}
    if set(assignment) != set(dates):
        raise ValueError("OOF fold assignment lost or added a training date")

    a.output_dir.mkdir(parents=True, exist_ok=False)
    out = a.output_dir / "train_date_fold.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date_BJT", "split", "heldout_fold"])
        for date in sorted(assignment):
            w.writerow([date, "train", assignment[date]])
    report = {
        "state": "HUBEI_TRAIN_DATE_OOF_FOLDS_FROZEN_METADATA_ONLY",
        "source_split_sha256": EXPECTED_SPLIT_SHA,
        "fold_manifest_sha256": sha(out),
        "fold_count": FOLDS,
        "train_dates": len(dates),
        "dates_by_fold": {str(f): sum(v == f for v in assignment.values()) for f in range(FOLDS)},
        "month_fold_date_counts": month_counts,
        "validation_dates_used_for_folds": 0,
        "test_or_buffer_dates_read": 0,
        "assignment_rule": "within each calendar month, sort train BJT dates by SHA256(hubei-oof-datefold-v1|month|date), then choose a deterministic round-robin rotation that minimizes cumulative fold-count imbalance; dates only",
        "use_rule": "For each fold, train S and R only on other frozen train dates; select/checkpoint using the unchanged global validation dates only; predict held-out fold after training. Keep all sequence windows within one BJT date and global train split.",
    }
    (a.output_dir / "fold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
