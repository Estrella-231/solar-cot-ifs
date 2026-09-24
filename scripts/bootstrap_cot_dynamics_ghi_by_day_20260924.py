"""Paired validation-date bootstrap for the geometry24 COT/GHI screen.

Forecasts with the same initialization date move together. This estimates
validation-period sampling variability, not training-seed or test uncertainty.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("A0", "AH", "AI", "O_diag")
COMPARISONS = (("AH", "A0"), ("AI", "A0"), ("AI", "AH"),
               ("O_diag", "AI"), ("O_diag", "AH"), ("O_diag", "A0"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--paired", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260924)
    args = p.parse_args()
    if args.output.exists() or args.replicates < 100:
        raise RuntimeError("refuse overwrite or insufficient bootstrap replicates")
    metrics_path = args.paired / "paired_metrics.json"
    prediction_path = args.paired / "matched_validation_predictions.npz"
    source = json.loads(metrics_path.read_text())
    if (source["state"] != "COMPLETE_GEOMETRY24_GHI_PAIRED_VALIDATION" or
            source["test_used"] or source["reference"]["n_rows"] != 99849 or
            sha(prediction_path) != source["matched_predictions_sha256"] or
            tuple(sorted(source["arms"])) != tuple(sorted(ARMS))):
        raise RuntimeError("paired validation source contract drift")
    with np.load(prediction_path, allow_pickle=False) as z:
        row = z["pack_row"].astype(np.int64)
        station = z["station"].astype(np.int64)
        truth = z["observed_ghi"].astype(np.float64)
        forecasts = {arm: z[f"pred_ghi_{arm}"].astype(np.float64) for arm in ARMS}
    if (len(row) != 99849 or not np.array_equal(row, np.unique(row)) or
            not np.isin(station, (0, 1)).all() or not np.isfinite(truth).all() or
            not all(np.isfinite(value).all() for value in forecasts.values())):
        raise RuntimeError("invalid paired prediction rows")
    dates = np.empty(len(row), dtype="U10")
    pos = 0
    with args.rows.open(newline="", encoding="utf-8-sig") as stream:
        for index, item in enumerate(csv.DictReader(stream)):
            if pos < len(row) and index == row[pos]:
                if item["split"] != "validation" or int(item["station_index"]) != station[pos]:
                    raise RuntimeError("row/date/station mismatch")
                dates[pos] = item["init_time_bjt"][:10]
                pos += 1
    if pos != len(row):
        raise RuntimeError("validation row/date mapping incomplete")
    unique_dates, date_index = np.unique(dates, return_inverse=True)
    n_dates = len(unique_dates)
    counts = np.zeros((n_dates, 2), dtype=np.int64)
    np.add.at(counts, (date_index, station), 1)
    squared = {}
    for arm, pred in forecasts.items():
        value = np.zeros((n_dates, 2), dtype=np.float64)
        np.add.at(value, (date_index, station), (pred - truth) ** 2)
        squared[arm] = value
    original_counts = counts.sum(axis=0)
    original = {arm: float(np.sqrt(value.sum(axis=0) / original_counts).mean())
                for arm, value in squared.items()}
    for arm in ARMS:
        if abs(original[arm] - source["arms"][arm]["equal_station_rmse_wm2"]) > 1e-6:
            raise RuntimeError("date aggregation differs from saved paired metric")
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, n_dates, size=(args.replicates, n_dates))
    weights = np.zeros((args.replicates, n_dates), dtype=np.int16)
    np.add.at(weights, (np.repeat(np.arange(args.replicates), n_dates), draws.ravel()), 1)
    resampled_counts = weights @ counts
    if (resampled_counts <= 0).any():
        raise RuntimeError("bootstrap draw contains no station rows")
    samples = {arm: np.sqrt((weights @ value) / resampled_counts).mean(axis=1)
               for arm, value in squared.items()}
    comparisons = {}
    for first, second in COMPARISONS:
        delta = samples[first] - samples[second]
        comparisons[f"{first}_minus_{second}"] = {
            "point_equal_station_rmse_wm2": original[first] - original[second],
            "bootstrap_percentile_95_ci_wm2": np.quantile(delta, (.025, .975)).tolist(),
            "fraction_of_draws_below_zero": float((delta < 0).mean()),
        }
    report = {
        "state": "COMPLETE_EXPLORATORY_PAIRED_INIT_DATE_BOOTSTRAP",
        "test_used": False,
        "interpretation": "validation initialization-date resampling only; excludes training-seed and independent-test uncertainty",
        "dates": n_dates,
        "date_min": str(unique_dates[0]),
        "date_max": str(unique_dates[-1]),
        "validation_rows": len(row),
        "station_rows": original_counts.tolist(),
        "replicates": args.replicates,
        "seed": args.seed,
        "paired_metrics_sha256": sha(metrics_path),
        "paired_predictions_sha256": sha(prediction_path),
        "rows_sha256": sha(args.rows),
        "arms_equal_station_rmse_wm2": original,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"state": report["state"], "dates": n_dates,
                      "comparisons": comparisons}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
