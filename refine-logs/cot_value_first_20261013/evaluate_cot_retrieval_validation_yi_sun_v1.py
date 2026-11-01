#!/usr/bin/env python3
"""Independent physical-COT validation metrics for the monthly Hunan R."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


SCALES = {
    "station_pixel_8_8": (slice(8, 9), slice(8, 9)),
    "local_3x3_7_10": (slice(7, 10), slice(7, 10)),
    "station_patch_16x16": (slice(0, 16), slice(0, 16)),
}


def stats(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict:
    p, y = pred[valid], ref[valid]
    if p.size == 0:
        return {"n_pixels": 0, "rmse": None, "mae": None, "bias": None, "r2": None}
    err = p - y
    den = np.square(y - y.mean()).sum(dtype=np.float64)
    return {
        "n_pixels": int(p.size),
        "rmse": float(np.sqrt(np.square(err).mean(dtype=np.float64))),
        "mae": float(np.abs(err).mean(dtype=np.float64)),
        "bias": float(err.mean(dtype=np.float64)),
        "r2_sse": float(1.0 - np.square(err).sum(dtype=np.float64) / den) if den > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--clp-reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")

    with np.load(a.predictions, allow_pickle=False) as z:
        ids = z["sample_indices"].astype(np.int64)
        pred_log = z["pred_log1p_cot"].astype(np.float64)
        ref = z["reference_cot"].astype(np.float64)
        ref_mask = z["mask"].astype(bool)
    with np.load(a.clp_reference, allow_pickle=False) as z:
        clp_ids = z["sample_indices"].astype(np.int64)
        stations = z["station"].astype(str)
        target_utc = z["target_time_utc"].astype(str)
        clp_mask = z["common_mask"].astype(bool)

    if pred_log.shape != ref.shape or pred_log.shape != ref_mask.shape:
        raise ValueError("prediction/reference/mask shapes differ")
    if pred_log.ndim != 4 or pred_log.shape[1:] != (1, 16, 16):
        raise ValueError(f"unexpected model output shape: {pred_log.shape}")
    if not (np.array_equal(ids, clp_ids) and len(ids) == len(stations) == len(target_utc)):
        raise ValueError("sample keys/order do not match the CLP validation sidecar")
    if not np.array_equal(ref_mask[:, 0], clp_mask[:, 0]):
        raise ValueError("CPP-COT and CLP common-support masks differ")
    if len(set(zip(stations.tolist(), target_utc.tolist()))) != len(ids):
        raise ValueError("duplicate (station,target_time) evaluation keys")

    # Model head outputs physical log1p(COT); invert once, in float64.
    with np.errstate(over="ignore", invalid="ignore"):
        pred = np.expm1(pred_log)
    valid = ref_mask
    if not np.isfinite(ref[valid]).all() or not np.isfinite(pred[valid]).all():
        raise ValueError("nonfinite physical COT on reference-valid support")
    if np.any((ref[valid] < 0) | (ref[valid] > 100)):
        raise ValueError("CPP reference outside declared [0,100] physical range")

    scale_results: dict[str, dict] = {}
    for scale, (ys, xs) in SCALES.items():
        p, y, m = pred[:, 0, ys, xs], ref[:, 0, ys, xs], valid[:, 0, ys, xs]
        by_station = {}
        for station in ("sili", "zhujia"):
            sel = stations == station
            by_station[station] = stats(p[sel], y[sel], m[sel])
        macro = {
            k: float(np.mean([by_station[s][k] for s in ("sili", "zhujia")]))
            if all(by_station[s][k] is not None for s in ("sili", "zhujia")) else None
            for k in ("rmse", "mae", "bias", "r2_sse")
        }
        scale_results[scale] = {
            "by_station": by_station,
            "two_station_equal_weight": macro,
            "pooled_pixels_diagnostic": stats(p, y, m),
        }

    # Cluster bootstrap by BJT target date; preserve station and pixel pairing.
    bjt_dates = []
    for t in target_utc:
        stamp = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc)
        bjt_dates.append((stamp + timedelta(hours=8)).date().isoformat())
    bjt_dates = np.asarray(bjt_dates)
    dates = np.unique(bjt_dates)
    rng = np.random.default_rng(a.seed)
    boot = {s: {k: [] for k in ("rmse", "mae", "bias")} for s in SCALES}
    for _ in range(a.bootstrap):
        draw = rng.choice(dates, size=len(dates), replace=True)
        counts = {d: int(np.count_nonzero(draw == d)) for d in dates}
        for scale, (ys, xs) in SCALES.items():
            p, y, m = pred[:, 0, ys, xs], ref[:, 0, ys, xs], valid[:, 0, ys, xs]
            site_metrics = {}
            for station in ("sili", "zhujia"):
                row_weights = np.asarray(
                    [counts[d] if st == station else 0 for st, d in zip(stations, bjt_dates)],
                    dtype=np.int64,
                )
                selected = (row_weights > 0)
                if not selected.any():
                    site_metrics[station] = None
                    continue
                wm = m[selected]
                weights = np.broadcast_to(row_weights[selected, None, None], wm.shape)[wm]
                err = (p[selected] - y[selected])[wm]
                val = p[selected][wm]
                truth = y[selected][wm]
                if weights.size == 0:
                    site_metrics[station] = None
                    continue
                total = weights.sum(dtype=np.float64)
                mse = np.sum(weights * np.square(err), dtype=np.float64) / total
                site_metrics[station] = {
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(np.sum(weights * np.abs(err), dtype=np.float64) / total),
                    "bias": float(np.sum(weights * err, dtype=np.float64) / total),
                }
            if all(site_metrics[s] is not None for s in ("sili", "zhujia")):
                for key in ("rmse", "mae", "bias"):
                    boot[scale][key].append(
                        float(np.mean([site_metrics[s][key] for s in ("sili", "zhujia")]))
                    )
    bootstrap_ci = {}
    for scale in SCALES:
        bootstrap_ci[scale] = {}
        for key, values in boot[scale].items():
            arr = np.asarray(values, dtype=np.float64)
            bootstrap_ci[scale][key] = {
                "valid_replicates": int(arr.size),
                "percentile_95_ci": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
                if arr.size else None,
            }

    strata = {}
    for name, condition in (
        ("reference_cot_lt_0_1", ref < 0.1),
        ("reference_cot_0_1_to_lt_30", (ref >= 0.1) & (ref < 30)),
        ("reference_cot_ge_30", ref >= 30),
    ):
        strata[name] = {}
        for station in ("sili", "zhujia"):
            sel = stations == station
            use = valid & condition
            strata[name][station] = stats(pred[sel], ref[sel], use[sel])

    out = {
        "state": "PASS_INDEPENDENT_PHYSICAL_COT_VALIDATION_RECOMPUTE",
        "split": "monthly_block_validation_only",
        "test_payloads_read": 0,
        "reference": "FY-4B CPP retrieval; not independent in-situ truth",
        "prediction_transform": "expm1(pred_log1p_cot) exactly once, float64",
        "sample_count": int(len(ids)),
        "station_sample_counts": {s: int(np.count_nonzero(stations == s)) for s in ("sili", "zhujia")},
        "unique_bjt_target_dates": int(len(dates)),
        "target_date_range_bjt": [str(dates[0]), str(dates[-1])],
        "common_support_pixels_16x16": int(valid.sum()),
        "possible_pixels_16x16": int(valid.size),
        "coverage_16x16": float(valid.sum() / valid.size),
        "prediction_physical_range_on_support": [float(pred[valid].min()), float(pred[valid].max())],
        "reference_physical_range_on_support": [float(ref[valid].min()), float(ref[valid].max())],
        "scale_definition": {
            "station_pixel_8_8": "station-centered crop center, zero-based [8,8]",
            "local_3x3_7_10": "crop rows/columns [7:10], retains each physical pixel",
            "station_patch_16x16": "native station-centered patch, not a regional full-field score",
        },
        "metrics": scale_results,
        "reference_cot_strata_16x16": strata,
        "bjt_target_date_cluster_bootstrap": {
            "replicates": int(a.bootstrap),
            "seed": int(a.seed),
            "estimator": "equal mean of station metrics, resampling target BJT dates with replacement",
            "confidence_intervals": bootstrap_ci,
        },
        "ssim": {
            "status": "NOT_COMPUTED",
            "reason": "train-only physical COT q99.9 scale is not bundled with the validation NPZ; local 3x3 is also below the fixed 11x11 window.",
        },
        "source_sha256": {
            "predictions": hashlib.sha256(a.predictions.read_bytes()).hexdigest(),
            "clp_reference": hashlib.sha256(a.clp_reference.read_bytes()).hexdigest(),
        },
        "limitations": [
            "CPP COT is a retrieval reference, not an independent physical truth.",
            "Only the previously available validation prediction bank is scored; it is not an untouched test result.",
            "The station-patch metric is not a regional full-image evaluation.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
