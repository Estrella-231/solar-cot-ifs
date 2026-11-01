#!/usr/bin/env python3
"""Audit predicted COT tails in the exact station-centred 3x3 GHI inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8-sig")))
    cot = np.load(a.pack / "cot_pred_3x3.npy", mmap_mode="r", allow_pickle=False)
    pcloud = np.load(a.pack / "p_cloud_3x3.npy", mmap_mode="r", allow_pickle=False)
    agri = np.load(a.pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(a.pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    ghi = np.load(a.pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear = np.load(a.pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    if cot.shape != (len(rows), 1, 3, 3):
        raise ValueError(f"unexpected COT shape {cot.shape} for {len(rows)} rows")
    if agri.shape != (len(rows), 13, 3, 3):
        raise ValueError(f"unexpected AGRI shape {agri.shape} for {len(rows)} rows")
    x = np.asarray(cot[:, 0], dtype=np.float64).reshape(len(rows), 9)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("non-finite or negative station COT values")

    summary = {}
    row_group = {(r["station"], r["split"]): [] for r in rows}
    for i, r in enumerate(rows):
        row_group[(r["station"], r["split"])].append(i)
    for station in sorted({r["station"] for r in rows}):
        summary[station] = {}
        train_ix = np.asarray(row_group[(station, "train")], dtype=np.int64)
        train_log = np.log1p(x[train_ix])
        log_mean = float(train_log.mean(dtype=np.float64))
        log_std = max(float(train_log.std(dtype=np.float64)), 1e-6)
        for split in ("train", "val"):
            ix = np.asarray(row_group[(station, split)], dtype=np.int64)
            v = x[ix].reshape(-1)
            z = (np.log1p(v) - log_mean) / log_std
            patch_max = x[ix].max(axis=1)
            summary[station][split] = {
                "rows": int(ix.size), "pixel_values": int(v.size),
                "pixel_percentiles": {str(q): float(np.percentile(v, q)) for q in (50, 90, 95, 99, 99.9, 99.99)},
                "pixel_max": float(v.max(initial=0)),
                "pixel_counts_above": {str(t): int(np.count_nonzero(v > t)) for t in (100, 1000, 10000, 100000)},
                "patches_with_any_above": {str(t): int(np.count_nonzero(patch_max > t)) for t in (100, 1000, 10000, 100000)},
                "trainer_log1p_train_normalization": {"mean": log_mean, "std": log_std,
                    "normalized_max": float(z.max(initial=0)),
                    "pixel_count_abs_z_above_5": int(np.count_nonzero(np.abs(z) > 5.0)),
                    "pixel_count_abs_z_above_8": int(np.count_nonzero(np.abs(z) > 8.0))},
            }

    top = np.argsort(x.max(axis=1))[-20:][::-1]
    extreme_rows = []
    for i in top:
        if x[i].max() <= 1000:
            break
        loc = int(np.argmax(x[i]))
        extreme_rows.append({"row_index": int(i), "station": rows[i]["station"],
                             "split": rows[i]["split"], "time_bjt": rows[i]["time_bjt"],
                             "patch_max": float(x[i].max()),
                             "max_pixel_rc_in_3x3": [loc // 3, loc % 3],
                             "pixel_values": x[i].reshape(3, 3).tolist(),
                             "p_cloud_3x3": np.asarray(pcloud[i, 0], dtype=np.float64).tolist(),
                             "ghi_wm2": float(ghi[i]), "clear_sky_wm2": float(clear[i]),
                             "cosSOZ_3x3": np.asarray(geometry[i, 0], dtype=np.float64).tolist(),
                             "agri_channel_median_3x3": np.median(np.asarray(agri[i], dtype=np.float64), axis=(1, 2)).tolist()})
    zero_agri_rows = []
    for i, row in enumerate(rows):
        values = np.asarray(agri[i], dtype=np.float64)
        if np.count_nonzero(values) == 0:
            zero_agri_rows.append({"row_index": i, "station": row["station"], "split": row["split"],
                                   "time_bjt": row["time_bjt"], "cot_max_3x3": float(x[i].max()),
                                   "p_cloud_min_3x3": float(np.min(pcloud[i])),
                                   "p_cloud_max_3x3": float(np.max(pcloud[i])),
                                   "ghi_wm2": float(ghi[i]), "clear_sky_wm2": float(clear[i]),
                                   "cosSOZ_min_3x3": float(np.min(geometry[i, 0])),
                                   "cosSOZ_max_3x3": float(np.max(geometry[i, 0]))})
    report = {"state": "PASS_STATION_3X3_COT_TAIL_AUDIT_TRAIN_VAL_ONLY",
              "pack": str(a.pack), "rows": len(rows), "shape": list(cot.shape),
              "by_station_split": summary, "largest_rows_over_1000": extreme_rows,
              "all_zero_agri_rows_in_station_3x3": zero_agri_rows,
              "test_payloads_read": 0, "buffer_payloads_read": 0}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
