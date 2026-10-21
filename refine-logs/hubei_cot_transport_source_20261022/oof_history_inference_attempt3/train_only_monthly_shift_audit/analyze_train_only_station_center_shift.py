#!/usr/bin/env python3
"""Audit train-only full-R vs date-held-out-R shift at Hubei station pixels.

Only train rows' COT array values are indexed. Validation rows are used for
key/split metadata checks only; validation values, targets, test, and buffer
payloads are never accessed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-bank", type=Path, required=True)
    ap.add_argument("--oof-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    report = json.loads((a.full_bank / "BANK_REPORT.json").read_text(encoding="utf-8"))
    assert report.get("test_payloads_read") == 0
    assert report.get("train_predictions_are_oof") is False
    cache_report = json.loads((a.oof_root.parents[2] / "cache_attempt5/cache/cache_report.json").read_text(encoding="utf-8"))
    station_names = cache_report["stations"]["names"]
    centers = [tuple(map(int, xy)) for xy in cache_report["stations"]["local_centers_row_col"]]
    assert len(station_names) == len(centers) == 3

    bank_rows = read_rows(a.full_bank / "rows.csv")
    assert len(bank_rows) == 22061
    bank_index: dict[str, int] = {}
    train_keys = set()
    for i, row in enumerate(bank_rows):
        key = row["time_bjt"]
        split = row["split"].strip().lower()
        assert split in {"train", "val"}
        assert key not in bank_index
        bank_index[key] = i
        if split == "train":
            train_keys.add(key)
    assert len(train_keys) == 20206

    oof_index: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    fold_counts = {}
    for fold in range(5):
        fold_dir = a.oof_root / f"fold_{fold}"
        rows = read_rows(fold_dir / "rows.csv")
        cot = np.load(fold_dir / "cot_oof.npy", mmap_mode="r", allow_pickle=False)
        day = np.load(fold_dir / "day_mask.npy", mmap_mode="r", allow_pickle=False)
        assert cot.shape == day.shape
        assert cot.ndim == 3 and cot.shape[1:] == (16, 16)
        assert len(rows) == cot.shape[0]
        fold_counts[str(fold)] = len(rows)
        for i, row in enumerate(rows):
            key = row["time_bjt"]
            assert key in train_keys and key not in oof_index, key
            oof_index[key] = (cot, day, i, fold)
    assert set(oof_index) == train_keys

    full = np.load(a.full_bank / "cot_pred_trainval_full.npy", mmap_mode="r", allow_pickle=False)
    assert full.shape in {(22061, 16, 16), (22061, 1, 16, 16)}, full.shape

    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "records": 0, "dates": set(), "day_records": 0,
        "common": 0, "full_sum": 0.0, "oof_sum": 0.0,
        "diff_sum": 0.0, "diff_sq_sum": 0.0,
        "full_gt100": 0, "oof_gt100": 0,
        "full_finite_day": 0, "oof_finite_day": 0,
    })
    for key in sorted(train_keys):
        bank_i = bank_index[key]
        cot, day, oof_i, _fold = oof_index[key]
        month = key[:6]
        for station, (r, c) in zip(station_names, centers):
            g = groups[(month, station)]
            g["records"] += 1
            g["dates"].add(key[:8])
            day_flag = bool(day[oof_i, r, c] > 0.5)
            if not day_flag:
                continue
            g["day_records"] += 1
            f = float(full[bank_i, 0, r, c] if full.ndim == 4 else full[bank_i, r, c])
            o = float(cot[oof_i, r, c])
            if math.isfinite(f):
                g["full_finite_day"] += 1
                g["full_gt100"] += int(f > 100.0)
            if math.isfinite(o):
                g["oof_finite_day"] += 1
                g["oof_gt100"] += int(o > 100.0)
            if not (math.isfinite(f) and math.isfinite(o) and 0.0 <= f <= 100.0 and 0.0 <= o <= 100.0):
                continue
            d = o - f
            g["common"] += 1
            g["full_sum"] += f
            g["oof_sum"] += o
            g["diff_sum"] += d
            g["diff_sq_sum"] += d * d

    out_rows = []
    for (month, station), g in sorted(groups.items()):
        n = g["common"]
        out_rows.append({
            "month_bjt": month,
            "station": station,
            "train_dates": len(g["dates"]),
            "daytime_records": g["day_records"],
            "common_inrange_records": n,
            "full_r_center_mean_cot": g["full_sum"] / n if n else "",
            "oof_r_center_mean_cot": g["oof_sum"] / n if n else "",
            "oof_minus_full_center_mean_cot": g["diff_sum"] / n if n else "",
            "center_input_shift_rmse_cot": math.sqrt(g["diff_sq_sum"] / n) if n else "",
            "full_r_gt100_day_pct": 100 * g["full_gt100"] / g["full_finite_day"] if g["full_finite_day"] else "",
            "oof_r_gt100_day_pct": 100 * g["oof_gt100"] / g["oof_finite_day"] if g["oof_finite_day"] else "",
        })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(json.dumps({
        "state": "PASS_TRAIN_ONLY_CENTER_SOURCE_SHIFT",
        "full_bank_train_rows_indexed": len(train_keys),
        "oof_train_rows": len(oof_index),
        "oof_fold_rows": fold_counts,
        "stations": dict(zip(station_names, centers)),
        "validation_prediction_values_accessed": 0,
        "cpp_target_or_qc_payloads_read": 0,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "output_csv": str(a.out),
    }, indent=2))


if __name__ == "__main__":
    main()
