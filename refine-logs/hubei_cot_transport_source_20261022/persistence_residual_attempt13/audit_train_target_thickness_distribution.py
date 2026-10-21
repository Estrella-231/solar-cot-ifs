#!/usr/bin/env python3
"""Audit COT thickness support in train-only forecast supervision.

The script reads only train rows from the already audited train/validation
station-patch cache. CPP values are treated as retrieval-reference labels.
It reports pixel exposure induced by the overlapping init/lead windows, which
matches the equal-lead masked training objective more closely than counting
unique frames alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


BINS = (("near_zero_lt1", 0.0, 1.0), ("thin_1_10", 1.0, 10.0),
        ("moderate_10_30", 10.0, 30.0), ("thick_ge30", 30.0, float("inf")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache
    report = json.loads((cache / "cache_report.json").read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_HUBEI_STATION_PATCH_COT_GEOMETRY_CACHE_TRAINVAL_ONLY":
        raise ValueError("unexpected cache state")
    if report.get("split", {}).get("test_payloads_read") != 0 or report.get("split", {}).get("buffer_payloads_read") != 0:
        raise ValueError("cache does not certify test/buffer closure")

    cot = np.load(cache / "cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(cache / "cpp_valid.npy", mmap_mode="r", allow_pickle=False)
    if cot.shape != valid.shape or cot.ndim != 4 or cot.shape[1:] != (1, 16, 16):
        raise ValueError(f"unexpected CPP arrays: cot={cot.shape}, valid={valid.shape}")

    row_counts: dict[tuple[int, int], int] = Counter()
    target_dates: set[str] = set()
    train_rows = 0
    with (cache / "forecast_pairs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            train_rows += 1
            idx, lead = int(row["target_index"]), int(row["lead_minutes"])
            if idx < 0 or idx >= cot.shape[0]:
                raise IndexError(f"target index outside CPP bank: {idx}")
            row_counts[(idx, lead)] += 1
            target_dates.add(row["target_time_bjt"][:8])

    if train_rows != int(report.get("forecast_pairs", {}).get("train", {}).get("pairs", -1)):
        raise ValueError("train pair count disagrees with the cache report")

    # Cache each distinct frame's valid-pixel histogram once, then weight it by
    # the number of train init/lead examples in which it is supervised.
    frame_hist: dict[int, np.ndarray] = {}
    for idx, _ in row_counts:
        x = np.asarray(cot[idx, 0], dtype=np.float64)
        m = np.asarray(valid[idx, 0], dtype=bool) & np.isfinite(x) & (x >= 0.0) & (x <= 100.0)
        frame_hist[idx] = np.array([np.count_nonzero(m & (x >= lo) & (x < hi))
                                    for _, lo, hi in BINS], dtype=np.int64)

    per_lead: dict[str, dict[str, int | float]] = {}
    total = np.zeros(len(BINS), dtype=np.int64)
    for lead in sorted({lead for _, lead in row_counts}):
        counts = np.zeros(len(BINS), dtype=np.int64)
        pair_n = 0
        for (idx, this_lead), multiplicity in row_counts.items():
            if this_lead == lead:
                counts += frame_hist[idx] * multiplicity
                pair_n += multiplicity
        total += counts
        denom = int(counts.sum())
        per_lead[str(lead)] = {
            "train_init_lead_pairs": pair_n,
            "valid_cpp_target_pixels": denom,
            **{name: int(count) for (name, _, _), count in zip(BINS, counts)},
            **{f"fraction_{name}": (float(count / denom) if denom else None)
               for (name, _, _), count in zip(BINS, counts)},
        }

    total_n = int(total.sum())
    out = {
        "state": "TRAIN_ONLY_CPP_REFERENCE_THICKNESS_EXPOSURE_AUDIT",
        "scope": "Hubei shared 16x16 station patch; train forecast-pair pixel exposure",
        "cpp_role": "satellite retrieval reference, not independent physical truth",
        "weighting": "each valid target pixel counted once per train init/lead pair; losses are equal-lead masked log1p-COT MSE",
        "train_init_lead_pairs": train_rows,
        "unique_target_dates_bjt": len(target_dates),
        "valid_cpp_target_pixels_across_overlapping_pairs": total_n,
        "train_cpp_target_pixel_counts": {name: int(count) for (name, _, _), count in zip(BINS, total)},
        "train_cpp_target_pixel_fractions": {name: (float(count / total_n) if total_n else None)
                                              for (name, _, _), count in zip(BINS, total)},
        "per_lead": per_lead,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
