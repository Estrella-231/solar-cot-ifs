"""Descriptive monthly/solar-elevation analysis of frozen Hubei validation predictions.

This script never opens test/buffer data. It reuses saved validation predictions
and the common train/validation pack, preserves paired rows, and reports
station-equal / seed-equal RMSE summaries. Cloud classes are intentionally not
inferred from model-predicted COT.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
SEEDS = (42, 43, 44)


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"index", "split", "station", "time_bjt", "ghi_wm2"}.issubset(rows[0]):
        raise ValueError("Unexpected pack rows schema")
    return rows


def metrics(y: np.ndarray, p: np.ndarray):
    e = p.astype(np.float64) - y.astype(np.float64)
    return {
        "n": int(y.size),
        "rmse_wm2": float(np.sqrt(np.mean(e * e))),
        "mae_wm2": float(np.mean(np.abs(e))),
        "bias_wm2": float(np.mean(e)),
    }


def summarize(items):
    """Average station metrics equally within seed, then average seeds equally."""
    per_seed = {}
    for seed in SEEDS:
        vals = [x["rmse_wm2"] for x in items if x["seed"] == seed]
        per_seed[str(seed)] = float(np.mean(vals)) if vals else None
    valid = [v for v in per_seed.values() if v is not None]
    return {
        "mean_station_equal_seed_equal_rmse_wm2": float(np.mean(valid)) if valid else None,
        "seed_station_equal_rmse_wm2": per_seed,
        "station_seed_cells": items,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="persistent Hubei run root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    rows = read_rows(root / "pack" / "rows.csv")
    geometry = np.load(root / "pack" / "geometry.npy", mmap_mode="r")
    truth = np.load(root / "pack" / "ghi_wm2.npy", mmap_mode="r")
    if len(rows) != len(truth) or geometry.shape[0] != len(rows):
        raise ValueError("Pack row/label/geometry lengths differ")

    val_indices = [i for i, r in enumerate(rows) if r["split"] == "val"]
    if any(rows[i]["split"] != "val" for i in val_indices):
        raise AssertionError("Non-validation row selected")
    if not val_indices:
        raise ValueError("No validation rows")

    # geometry channel 0 is cos(SOZ); elevation = asin(cos(SOZ)).
    cos_sza = np.asarray(geometry[val_indices, 0, 1, 1], dtype=np.float64)
    elevation = np.degrees(np.arcsin(np.clip(cos_sza, -1.0, 1.0)))
    val_rows = [rows[i] for i in val_indices]
    y_all = np.asarray(truth[val_indices], dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise ValueError("Non-finite validation GHI label")

    groups = defaultdict(list)
    for j, r in enumerate(val_rows):
        date = r["time_bjt"][:6]  # YYYYMM, BJT
        groups[("month", date)].append(j)
        el = elevation[j]
        if 0 <= el < 15:
            key = "0-15deg"
        elif 15 <= el < 30:
            key = "15-30deg"
        elif 30 <= el < 45:
            key = "30-45deg"
        elif 45 <= el <= 90:
            key = "45-90deg"
        else:
            continue
        groups[("solar_elevation", key)].append(j)

    results = {}
    for (kind, label), positions in sorted(groups.items()):
        by_arm = {arm: [] for arm in ARMS}
        for station in STATIONS:
            station_positions = [j for j in positions if val_rows[j]["station"] == station]
            if not station_positions:
                continue
            global_indices = np.asarray([val_indices[j] for j in station_positions], dtype=np.int64)
            y = np.asarray(truth[global_indices], dtype=np.float64)
            for seed in SEEDS:
                for arm in ARMS:
                    pred_path = root / "runs" / f"seed{seed}" / station / f"{arm}_val_prediction_ghi_wm2.npy"
                    p_all = np.load(pred_path, mmap_mode="r")
                    station_val_global = [i for i, r in enumerate(rows) if r["split"] == "val" and r["station"] == station]
                    if p_all.shape[0] != len(station_val_global):
                        raise ValueError(f"Prediction length mismatch: {pred_path}")
                    position_map = {global_i: local_i for local_i, global_i in enumerate(station_val_global)}
                    p = np.asarray(p_all[[position_map[int(i)] for i in global_indices]], dtype=np.float64)
                    cell = {"station": station, "seed": seed, **metrics(y, p)}
                    by_arm[arm].append(cell)
        arm_summary = {arm: summarize(by_arm[arm]) for arm in ARMS}
        deltas = {}
        for left, right, name in (("AC", "A", "AC_minus_A"), ("AC", "AP", "AC_minus_AP")):
            # Pair within the same station and seed; a negative RMSE delta favors AC.
            lm = {(x["station"], x["seed"]): x["rmse_wm2"] for x in by_arm[left]}
            rm = {(x["station"], x["seed"]): x["rmse_wm2"] for x in by_arm[right]}
            pairs = [{"station": s, "seed": seed, "delta_rmse_wm2": lm[(s, seed)] - rm[(s, seed)]}
                     for s, seed in sorted(set(lm) & set(rm))]
            deltas[name] = {
                "mean_station_equal_seed_equal_delta_rmse_wm2": float(np.mean([x["delta_rmse_wm2"] for x in pairs])) if pairs else None,
                "paired_station_seed_cells": pairs,
            }
        results[f"{kind}:{label}"] = {
            "n_validation_rows": len(positions),
            "n_by_station": {s: sum(val_rows[j]["station"] == s for j in positions) for s in STATIONS},
            "arms": arm_summary,
            "paired_deltas": deltas,
        }

    out = {
        "state": "COMPLETE_DESCRIPTIVE_HUBEI_VALIDATION_MONTH_SOLAR_ELEVATION_SUBGROUPS",
        "scope": "saved train/validation pack and saved validation predictions only",
        "test_or_buffer_payloads_read": 0,
        "selection_warning": "best epochs were selected on validation; subgroup results are exploratory and selection-affected, not independent confirmation",
        "cloud_strata": "omitted: independent CPP/CLP cloud labels are unavailable for the June-July extension; predicted COT was not used to define weather classes",
        "unit_contract": "original engineering W/m2 operating convention; sensor calibration/export metadata not independently verified",
        "solar_elevation_bins_deg": [0, 15, 30, 45, 90],
        "aggregation": "RMSE per station, equal station weight within each seed, then equal seed weight; AC deltas paired by station and seed",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "groups": list(results), "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
