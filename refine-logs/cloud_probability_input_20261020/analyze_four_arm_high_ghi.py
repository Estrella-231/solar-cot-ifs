#!/usr/bin/env python3
"""FP64 high-GHI paired error-mass diagnostic from saved validation predictions."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ARMS = ("M00", "M10", "M01", "M11")
STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=800.0)
    args = ap.parse_args()
    run = args.run
    score = json.loads((run / "score_summary_fp64.json").read_text(encoding="utf-8"))
    if score.get("state") != "COMPLETE_FP64_PAIRED_SCORE" or score.get("test_payloads_read") != 0:
        raise RuntimeError("scored validation/test provenance gate failed")
    rows = read_rows(run / "validation_rows.csv")
    inputs = np.load(run / "validation_inputs.npz", allow_pickle=False)
    ids_all = np.asarray(inputs["pack_index"], dtype=np.int64)
    if len(rows) != len(ids_all) or len(rows) != score["sample_count"]:
        raise RuntimeError("validation row/input support mismatch")
    truth_by_id = {int(i): float(np.float32(float(r["ghi"]))) for i, r in zip(ids_all, rows)}
    station_by_id = {int(i): r["station"] for i, r in zip(ids_all, rows)}
    high_by_station = {
        s: sorted(i for i in truth_by_id if station_by_id[i] == s and truth_by_id[i] >= args.threshold)
        for s in STATIONS
    }
    if any(not v for v in high_by_station.values()):
        raise RuntimeError("high-GHI subset is empty for a station")
    block_rows = read_rows(run / "date_blocks.csv")
    blocks = [[x for x in r["bjt_dates"].split(";") if x] for r in block_rows]
    date_to_block = {d: bi for bi, dates in enumerate(blocks) for d in dates}
    block_for_id = {}
    for i, r in zip(ids_all, rows):
        bjt_date = (datetime.fromisoformat(r["target_time_utc"]) + timedelta(hours=8)).date().isoformat()
        if bjt_date not in date_to_block:
            raise RuntimeError(f"validation date is absent from frozen block list: {bjt_date}")
        block_for_id[int(i)] = date_to_block[bjt_date]

    per_run = []
    paired_delta_sse = []
    base_sse_total = 0.0
    joint_sse_total = 0.0
    arm_stats = {arm: [] for arm in ARMS}
    block_sse = {}
    block_n = {}
    for seed in SEEDS:
        for station in STATIONS:
            wanted = np.asarray(high_by_station[station], dtype=np.int64)
            preds = {}
            y = None
            for arm in ARMS:
                z = np.load(run / f"seed{seed}" / station / arm / "validation_predictions.npz", allow_pickle=False)
                ids = np.asarray(z["pack_index"], dtype=np.int64)
                pred = np.asarray(z["prediction"], dtype=np.float64)
                truth = np.asarray(z["truth"], dtype=np.float64)
                where = np.searchsorted(ids, wanted)
                if np.any(where >= len(ids)) or not np.array_equal(ids[where], wanted):
                    raise RuntimeError(f"high-GHI keys missing for {seed}/{station}/{arm}")
                yy = truth[where]
                if y is None:
                    y = yy
                elif not np.array_equal(y, yy):
                    raise RuntimeError("paired arms have different high-GHI labels")
                preds[arm] = pred[where]
                err = pred[where] - yy
                arm_stats[arm].append({
                    "seed": seed, "station": station, "n": int(len(yy)),
                    "rmse": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
                    "mae": float(np.mean(np.abs(err), dtype=np.float64)),
                    "mbe": float(np.mean(err, dtype=np.float64)),
                })
                selected_blocks = np.asarray([block_for_id[int(i)] for i in wanted], dtype=np.int64)
                block_sse[(seed, station, arm)] = np.bincount(
                    selected_blocks, weights=err * err, minlength=len(blocks)).astype(np.float64)
                block_n[(seed, station, arm)] = np.bincount(selected_blocks, minlength=len(blocks)).astype(np.float64)
            base_err = preds["M10"] - y
            joint_err = preds["M11"] - y
            d_sse = joint_err * joint_err - base_err * base_err
            paired_delta_sse.extend(d_sse.tolist())
            base_sse_total += float(np.sum(base_err * base_err, dtype=np.float64))
            joint_sse_total += float(np.sum(joint_err * joint_err, dtype=np.float64))

    for arm in ARMS:
        per_run.append({
            "arm": arm,
            "equal_station_equal_seed_RMSE_Wm2": float(np.mean([r["rmse"] for r in arm_stats[arm]])),
            "equal_station_equal_seed_MAE_Wm2": float(np.mean([r["mae"] for r in arm_stats[arm]])),
            "equal_station_equal_seed_MBE_Wm2": float(np.mean([r["mbe"] for r in arm_stats[arm]])),
        })
    rng = np.random.default_rng(20261020)
    bootstrap_n = 2000
    accepted = []
    accepted_n = 0
    attempts = 0
    while accepted_n < bootstrap_n:
        batch_n = bootstrap_n - accepted_n
        candidate = rng.integers(0, len(blocks), size=(batch_n, len(blocks)), endpoint=False)
        valid = np.ones(batch_n, dtype=bool)
        for seed in SEEDS:
            for station in STATIONS:
                valid &= block_n[(seed, station, "M10")][candidate].sum(axis=1) > 0
        if valid.any():
            kept = candidate[valid][:bootstrap_n - accepted_n]
            accepted.append(kept)
            accepted_n += len(kept)
        attempts += batch_n
        if attempts > bootstrap_n * 1000:
            raise RuntimeError("too few valid high-GHI bootstrap replicates")
    draws = np.concatenate(accepted, axis=0)[:bootstrap_n]
    boot_arm_rmse = {}
    for arm in ("M10", "M11"):
        runs = []
        for seed in SEEDS:
            for station in STATIONS:
                bs_sse = block_sse[(seed, station, arm)][draws].sum(axis=1, dtype=np.float64)
                bs_n = block_n[(seed, station, arm)][draws].sum(axis=1, dtype=np.float64)
                if np.any(bs_n <= 0):
                    raise RuntimeError("empty high-GHI bootstrap support")
                runs.append(np.sqrt(bs_sse / bs_n))
        boot_arm_rmse[arm] = np.mean(np.stack(runs, axis=0), axis=0)
    high_delta_ci = np.percentile(boot_arm_rmse["M11"] - boot_arm_rmse["M10"], [2.5, 97.5])
    high_delta_rmse_point = float(np.mean([r["rmse"] for r in arm_stats["M11"]]) -
                                  np.mean([r["rmse"] for r in arm_stats["M10"]]))
    delta = np.asarray(paired_delta_sse, dtype=np.float64)
    summary = {
        "state": "COMPLETE_HIGH_GHI_DESCRIPTIVE",
        "evaluation_scope": "same-time frozen monthly validation; descriptive subset only",
        "threshold_GHI_Wm2": args.threshold,
        "threshold_applied_to": "observed 15-minute interval-mean GHI",
        "unique_rows": int(sum(len(v) for v in high_by_station.values())),
        "station_rows": {s: len(v) for s, v in high_by_station.items()},
        "seed_station_prediction_pairs": int(len(delta)),
        "arm_metrics_equal_station_equal_seed": per_run,
        "primary_paired_M11_minus_M10_RMSE": {
            "delta_RMSE_Wm2_candidate_minus_baseline": high_delta_rmse_point,
            "date_block_95pct_CI_delta_Wm2": [float(x) for x in high_delta_ci],
            "bootstrap_replicates": bootstrap_n,
            "bootstrap_seed": 20261020,
            "bootstrap_block_count": len(blocks),
            "bootstrap_draws_rejected_for_empty_station_seed_support": int(attempts - accepted_n),
        },
        "primary_paired_M11_minus_M10_SSE": {
            "M10_total_SSE_W2m4": base_sse_total,
            "M11_total_SSE_W2m4": joint_sse_total,
            "delta_SSE_W2m4_candidate_minus_baseline": float(np.sum(delta, dtype=np.float64)),
            "relative_SSE_reduction_percent": float(100.0 * (base_sse_total - joint_sse_total) / base_sse_total),
            "mean_delta_SSE_per_seed_station_row_W2m4": float(np.mean(delta)),
            "improved_pairs_delta_SSE_below_zero": int(np.count_nonzero(delta < 0)),
            "worsened_pairs_delta_SSE_above_zero": int(np.count_nonzero(delta > 0)),
            "unchanged_pairs": int(np.count_nonzero(delta == 0)),
        },
        "per_run": arm_stats,
        "test_payloads_read": 0,
    }
    out = run / "high_ghi_ge_800_fp64.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
