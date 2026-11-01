#!/usr/bin/env python3
"""Metadata-only feasibility audit for date-grouped OOF forecast inputs.

Reads only the existing forecast-bank index rows marked train. It does not
open imagery, GHI/CPP labels, validation, test, or buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fold_dates(dates: set[str], k: int, salt: str) -> dict[str, int]:
    """Balance whole BJT dates among folds within each calendar month."""
    if k < 2:
        raise ValueError("at least two folds are required")
    months: dict[str, list[str]] = defaultdict(list)
    for day in dates:
        months[day[:7]].append(day)
    out: dict[str, int] = {}
    for month, days in months.items():
        ordered = sorted(days, key=lambda d: hashlib.sha256(f"{salt}:{k}:{d}".encode()).hexdigest())
        for i, day in enumerate(ordered):
            out[day] = i % k
    return out


def sequence_frames(seq_id: str) -> list[str]:
    if not seq_id.startswith("hunan_"):
        raise ValueError(f"unexpected sequence id: {seq_id}")
    start = datetime.strptime(seq_id[6:], "%Y%m%d%H%M")
    return [(start + timedelta(minutes=15 * j)).strftime("%Y%m%d%H%M") for j in range(24)]


def bjt_day(utc_stamp: str) -> str:
    """Bank seq_id timestamps and frame stamps are UTC; fold by BJT date."""
    return (datetime.strptime(utc_stamp, "%Y%m%d%H%M") + timedelta(hours=8)).date().isoformat()


def audit(index_path: Path, ks=(2, 3, 5), salt="future_ghi_oof_20261019") -> dict:
    with index_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == "train"]
    ids = [r["seq_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate training sequence IDs")
    starts = {seq: datetime.strptime(seq[6:], "%Y%m%d%H%M") for seq in ids}
    date_labels = {(d + timedelta(hours=8)).date().isoformat() for d in starts.values()}
    summaries = {}
    for k in ks:
        assignment = fold_dates(date_labels, k, salt)
        per_fold = []
        oof_ids = set()
        for fold in range(k):
            heldout, train, purged = [], [], []
            held_frames, train_frames = set(), set()
            for seq in ids:
                frames = sequence_frames(seq)
                labels = [assignment.get(bjt_day(x), -1) for x in frames]
                frame_set = set(frames)
                if all(x == fold for x in labels):
                    heldout.append(seq); held_frames.update(frame_set); oof_ids.add(seq)
                elif all(x >= 0 and x != fold for x in labels):
                    train.append(seq); train_frames.update(frame_set)
                else:
                    purged.append(seq)
            overlap = held_frames & train_frames
            if overlap:
                raise AssertionError(f"fold {fold} has {len(overlap)} train/heldout frame overlaps")
            per_fold.append({"fold": fold, "train_sequences": len(train),
                             "heldout_oof_sequences": len(heldout),
                             "purged_boundary_sequences": len(purged),
                             "train_unique_frames": len(train_frames),
                             "heldout_unique_frames": len(held_frames),
                             "train_heldout_frame_intersection": 0})
        summaries[str(k)] = {
            "folds": per_fold,
            "oof_sequences_with_one_prediction": len(oof_ids),
            "oof_coverage_fraction": len(oof_ids) / len(ids),
            "fold_assignment": "whole BJT dates; month-stratified deterministic hash; any 24-frame sequence touching mixed folds is purged",
        }
    return {
        "state": "COMPLETE_TRAIN_ONLY_OOF_FOLD_FEASIBILITY_AUDIT",
        "bank_index_sha256": sha(index_path),
        "source_split": "train rows only from the completed train/validation forecast bank",
        "source_train_sequences": len(ids),
        "source_train_dates": len(date_labels),
        "fold_salt": salt,
        "candidate_folds": summaries,
        "metadata_only": True,
        "imagery_payload_opened": False,
        "ghi_or_cpp_labels_opened": False,
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "buffer_payload_opened": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = audit(a.index)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "sequences": result["source_train_sequences"],
                      "candidate_folds": {k: {"oof": v["oof_sequences_with_one_prediction"],
                                               "coverage": round(v["oof_coverage_fraction"], 4),
                                               "fold_train": [f["train_sequences"] for f in v["folds"]]}
                                           for k, v in result["candidate_folds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
