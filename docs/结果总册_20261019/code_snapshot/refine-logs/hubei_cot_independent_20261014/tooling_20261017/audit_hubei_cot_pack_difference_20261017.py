#!/usr/bin/env python3
"""Explain the COT-only difference between old and rebuilt train/val packs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-pack", type=Path, required=True)
    p.add_argument("--old-pack", type=Path, required=True)
    p.add_argument("--new-features", type=Path, required=True)
    p.add_argument("--new-feature-rows", type=Path, required=True)
    p.add_argument("--new-feature-report", type=Path, required=True)
    p.add_argument("--new-r-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    new_rows = read_rows(a.new_pack / "rows.csv")
    old_rows = read_rows(a.old_pack / "rows.csv")
    if new_rows != old_rows:
        raise ValueError("Old and rebuilt pack row tables are not exactly equal")
    new_cot = np.load(a.new_pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    old_cot = np.load(a.old_pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    feature_rows = read_rows(a.new_feature_rows)
    features = np.load(a.new_features, mmap_mode="r", allow_pickle=False)
    if new_cot.shape != old_cot.shape or new_cot.shape != (len(new_rows), 1, 3, 3):
        raise ValueError(f"Unexpected COT shapes: {new_cot.shape}, {old_cot.shape}")
    index_by_time = {r["time_bjt"]: i for i, r in enumerate(feature_rows)}
    if len(index_by_time) != len(feature_rows):
        raise ValueError("Duplicate feature timestamp")
    diffs: Counter[str] = Counter()
    n_diff = 0
    max_abs = 0.0
    candidates: list[dict[str, object]] = []
    feature_consistent = True
    for i, row in enumerate(new_rows):
        station = row["station"]
        si = ("Niushou", "Wangzhai", "Mizhuang").index(station)
        fi = index_by_time.get(row["time_bjt"])
        if fi is None:
            raise ValueError(f"Pack time missing from feature bank: {row['time_bjt']}")
        rebuilt_from_feature = np.asarray(features[fi, si, 16], dtype=np.float32)
        current = np.asarray(new_cot[i, 0], dtype=np.float32)
        if not np.array_equal(current, rebuilt_from_feature):
            feature_consistent = False
        delta = np.abs(current.astype(np.float64) - np.asarray(old_cot[i, 0], dtype=np.float64))
        changed = delta != 0
        n = int(np.count_nonzero(changed))
        n_diff += n
        max_abs = max(max_abs, float(delta.max(initial=0)))
        if n:
            diffs[f"{row['split']}:{station}"] += n
            if np.any(changed):
                loc = np.unravel_index(int(np.argmax(delta)), delta.shape)
                candidates.append({"split": row["split"], "station": station, "time_bjt": row["time_bjt"],
                                   "different_values_in_3x3": n, "max_abs_difference": float(delta.max()),
                                   "max_difference_cell": [int(loc[0]), int(loc[1])]})

    new_pack_report = json.loads((a.new_pack / "pack_report.json").read_text(encoding="utf-8"))
    old_pack_report = json.loads((a.old_pack / "pack_report.json").read_text(encoding="utf-8"))
    feature_report = json.loads(a.new_feature_report.read_text(encoding="utf-8"))
    r_report = json.loads(a.new_r_report.read_text(encoding="utf-8"))
    result = {
        "state": "HUBEI_COT_PACK_DIFFERENCE_AUDIT_TRAINVAL_TEST_CLOSED",
        "rows_exact_equal": True, "row_count": len(new_rows),
        "old_cot_sha256": sha(a.old_pack / "cot.npy"),
        "rebuilt_cot_sha256": sha(a.new_pack / "cot.npy"),
        "n_changed_cot_values": n_diff,
        "total_cot_values": int(new_cot.size),
        "max_abs_difference": max_abs,
        "changed_values_by_split_station": dict(diffs),
        "rebuilt_pack_matches_rebuilt_feature_bank": feature_consistent,
        "new_pack_report_r_bank_sha256": new_pack_report.get("r_bank_report_sha256"),
        "new_feature_report_r_bank_sha256": feature_report.get("r_bank_report_sha256"),
        "actual_new_r_report_sha256": sha(a.new_r_report),
        "new_r_report_prediction_sha256": r_report.get("prediction_sha256"),
        "new_pack_r_prediction_sha256": new_pack_report.get("r_prediction_sha256"),
        "new_pack_feature_report_sha256": new_pack_report.get("feature_report_sha256"),
        "actual_new_feature_report_sha256": sha(a.new_feature_report),
        "old_pack_r_checkpoint_sha256": old_pack_report.get("r_checkpoint_sha256"),
        "new_r_checkpoint_sha256": r_report.get("frozen_R_checkpoint_sha256"),
        "old_pack_split_sha256": old_pack_report.get("split_sha256"),
        "new_pack_split_sha256": new_pack_report.get("split_sha256"),
        "test_and_buffer_values_read": 0,
        "interpretation": "A COT mismatch is not silently normalized away; review source and inference contract before relying on prior COT-arm training metrics.",
        "largest_examples": sorted(candidates, key=lambda x: x["max_abs_difference"], reverse=True)[:20],
    }
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
