#!/usr/bin/env python3
"""Paired validation audit separating historical and forecast-COT effects.

Reads saved validation predictions only. Requires exact same sample keys and
truth across arms/seeds, reports station/lead metrics, and uses target-BJT-date
cluster bootstrap so overlapping init/lead rows are not treated as independent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        pred = np.asarray(z["prediction_ghi_wm2"], np.float64)
        truth = np.asarray(z["truth_ghi_wm2"], np.float64)
        keys = [json.loads(str(x)) for x in z["keys"]]
    if pred.shape != truth.shape or pred.ndim != 1 or len(keys) != len(pred):
        raise ValueError(f"invalid saved prediction bundle: {path}")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise ValueError(f"nonfinite values in {path}")
    canon = []
    for r in keys:
        canon.append((r["station"], r["init_time_utc"], r["target_time_utc"],
                      int(r["lead_minutes"]), int(r["ghi_row_index"])))
    if len(set(canon)) != len(canon):
        raise ValueError(f"duplicate forecast-instance keys in {path}")
    target_days = np.asarray([
        (datetime.strptime(k[2], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d")
        for k in canon
    ])
    leads = np.asarray([k[3] for k in canon], np.int16)
    return {"pred": pred, "truth": truth, "keys": canon,
            "target_days": target_days, "leads": leads}


def rmse(err: np.ndarray, weight=None) -> float:
    if weight is None:
        return float(np.sqrt(np.mean(np.square(err))))
    den = float(weight.sum())
    return float(np.sqrt(np.sum(weight * np.square(err)) / den)) if den else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261019)
    p.add_argument("--require-oof", action="store_true",
                   help="refuse scoring unless training forecast inputs are certified five-fold OOF")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    run_manifest_path = a.run_root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError("GHI run provenance manifest is missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("state") != "COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT":
        raise ValueError("GHI head run is not complete")
    train_oof = run_manifest.get("train_forecast_oof") is True
    if a.require_oof and not train_oof:
        raise ValueError("this audit requires out-of-fold training forecast inputs")
    if train_oof:
        if run_manifest.get("forecast_bank_oof_fold_count") != 5:
            raise ValueError("OOF run does not certify all five forecast-bank folds")
        if not run_manifest.get("forecast_bank_manifest_sha256"):
            raise ValueError("OOF forecast-bank manifest hash is missing")
    if set(a.seeds) - set(run_manifest.get("seeds", [])):
        raise ValueError("requested audit seeds are absent from run provenance")
    expected_runs = {(seed, station, arm) for seed in a.seeds
                     for station in ("sili", "zhujia") for arm in ("A", "AH", "AC")}
    recorded_runs = {(int(x["seed"]), x["station"], x["arm"])
                     for x in run_manifest.get("runs", [])
                     if "seed" in x and "station" in x and "arm" in x}
    if not expected_runs.issubset(recorded_runs):
        raise ValueError("run manifest lacks one or more requested station/seed/arm completions")
    data = {}
    for seed in a.seeds:
        for station in ("sili", "zhujia"):
            pair = {}
            for arm in ("A", "AH", "AC"):
                path = a.run_root / station / f"seed{seed}" / arm / "validation_predictions.npz"
                pair[arm] = load(path)
            A = pair["A"]
            for arm in ("AH", "AC"):
                if A["keys"] != pair[arm]["keys"]:
                    raise ValueError(f"paired sample keys differ: seed{seed}/{station}/{arm}")
                if not np.array_equal(A["truth"], pair[arm]["truth"]):
                    raise ValueError(f"paired truth arrays differ: seed{seed}/{station}/{arm}")
            if any(k[0] != station for k in A["keys"]):
                raise ValueError(f"station mismatch in saved keys: seed{seed}/{station}")
            data[(seed, station)] = pair

    # Ensure each station uses the same exact sample keys across seeds.
    for station in ("sili", "zhujia"):
        base = data[(a.seeds[0], station)]["A"]["keys"]
        for seed in a.seeds[1:]:
            if data[(seed, station)]["A"]["keys"] != base:
                raise ValueError(f"sample keys vary across seeds at {station}")
            for arm in ("AH", "AC"):
                if data[(seed, station)][arm]["keys"] != base:
                    raise ValueError(f"{arm} sample keys vary across seeds at {station}")

    days = sorted(set(data[(a.seeds[0], "sili")]["A"]["target_days"]) |
                  set(data[(a.seeds[0], "zhujia")]["A"]["target_days"]))
    if len(days) < 2:
        raise ValueError("too few target-date blocks for bootstrap")
    day_index = {d: i for i, d in enumerate(days)}
    rows = []
    for seed in a.seeds:
        for station in ("sili", "zhujia"):
            A = data[(seed, station)]["A"]
            leads = A["leads"]
            for arm in ("A", "AH", "AC"):
                item = data[(seed, station)][arm]
                err, pred = item["pred"] - item["truth"], item["pred"]
                rows.append({"seed": seed, "station": station, "arm": arm, "n": len(err),
                    "rmse_wm2": rmse(err), "mae_wm2": float(np.mean(np.abs(err))),
                    "bias_wm2": float(np.mean(err)),
                    "first_hour_rmse_wm2": rmse(err[leads <= 60]),
                    "one_to_two_hour_rmse_wm2": rmse(err[(leads > 60) & (leads <= 120)]),
                    "two_to_four_hour_rmse_wm2": rmse(err[leads > 120])})

    # Date-block bootstrap: resample BJT target dates jointly for both stations
    # and all seeds; negative pairwise delta favors the left-hand arm.
    rng = np.random.default_rng(a.bootstrap_seed)
    comparisons = (("AC", "A"), ("AC", "AH"), ("AH", "A"))
    draws = {f"{left}_minus_{right}": np.empty(a.bootstrap, np.float64)
             for left, right in comparisons}
    by_date = {}
    for seed in a.seeds:
        for station in ("sili", "zhujia"):
            pair = data[(seed, station)]
            day_ids = np.asarray([day_index[d] for d in pair["A"]["target_days"]], np.int32)
            for arm in ("A", "AH", "AC"):
                err = pair[arm]["pred"] - pair[arm]["truth"]
                by_date[(seed, station, arm)] = (
                    np.bincount(day_ids, weights=np.square(err), minlength=len(days)),
                    np.bincount(day_ids, minlength=len(days)).astype(np.float64),
                )
    point_by_station = {}
    for station in ("sili", "zhujia"):
        point_by_station[station] = {}
        for arm in ("A", "AH", "AC"):
            point_by_station[station][arm] = float(np.mean([
                next(r["rmse_wm2"] for r in rows if r["seed"] == seed and r["station"] == station and r["arm"] == arm)
                for seed in a.seeds
            ]))
    point = {arm: float(np.mean([point_by_station[s][arm] for s in ("sili", "zhujia")]))
             for arm in ("A", "AH", "AC")}
    for bootstrap_i in range(a.bootstrap):
        sampled = rng.integers(0, len(days), size=len(days))
        count = np.bincount(sampled, minlength=len(days)).astype(np.float64)
        for left, right in comparisons:
            station_deltas = []
            for station in ("sili", "zhujia"):
                seed_deltas = []
                for seed in a.seeds:
                    sse_l, n_l = by_date[(seed, station, left)]
                    sse_r, n_r = by_date[(seed, station, right)]
                    seed_deltas.append(float(np.sqrt(count @ sse_l / (count @ n_l)) -
                                             np.sqrt(count @ sse_r / (count @ n_r))))
                station_deltas.append(float(np.mean(seed_deltas)))
            draws[f"{left}_minus_{right}"][bootstrap_i] = float(np.mean(station_deltas))
    ci = {pair: np.quantile(values, [0.025, 0.975]).tolist()
          for pair, values in draws.items()}

    lead_summary = []
    for station in ("sili", "zhujia"):
        lead_values = data[(a.seeds[0], station)]["A"]["leads"]
        for lead in sorted(set(lead_values.tolist())):
            arm_rmse = {}
            for arm in ("A", "AH", "AC"):
                vals = []
                for seed in a.seeds:
                    pair = data[(seed, station)][arm]
                    err = pair["pred"] - pair["truth"]
                    vals.append(rmse(err[pair["leads"] == lead]))
                arm_rmse[arm] = float(np.mean(vals))
            lead_summary.append({"station": station, "lead_minutes": int(lead),
                                 **{f"{arm}_rmse_wm2": arm_rmse[arm] for arm in ("A", "AH", "AC")},
                                 "delta_AC_minus_A_wm2": arm_rmse["AC"] - arm_rmse["A"],
                                 "delta_AC_minus_AH_wm2": arm_rmse["AC"] - arm_rmse["AH"],
                                 "delta_AH_minus_A_wm2": arm_rmse["AH"] - arm_rmse["A"]})

    summary = {
        "state": "COMPLETE_PAIRED_FUTURE_GHI_COT_SOURCE_VALIDATION_AUDIT",
        "scope": ("validation screening with date-heldout OOF train forecasts and fixed full-train validation forecasts; independent test not accessed"
                  if train_oof else
                  "validation screening with in-sample train forecast inputs; not forecast-input generalization"),
        "train_forecast_oof": train_oof,
        "forecast_bank_manifest_sha256": run_manifest.get("forecast_bank_manifest_sha256"),
        "run_manifest_sha256": hashlib.sha256(run_manifest_path.read_bytes()).hexdigest(),
        "metric": "equal-station mean of seed-averaged RMSE; negative pairwise delta favors the first arm",
        "seeds": a.seeds, "stations": ["sili", "zhujia"], "target_bjt_date_blocks": len(days),
        "bootstrap_replicates": a.bootstrap, "bootstrap_seed": a.bootstrap_seed,
        "arms": {"A": "no COT", "AH": "observed-history COT only", "AC": "observed-history plus forecast-AGRI COT"},
        "primary_future_cot_contrast": "AC minus AH isolates the added predicted-future COT contribution",
        "pairwise_delta_rmse_wm2": {f"{left}_minus_{right}": point[left] - point[right]
                                     for left, right in comparisons},
        "equal_station_mean_seed_rmse_wm2": point,
        "pairwise_relative_rmse_change_pct": {
            f"{left}_minus_{right}": 100.0 * (point[left] - point[right]) / point[right]
            for left, right in comparisons},
        "date_block_bootstrap_95pct_ci_wm2": ci,
        "per_station_mean_seed_rmse": point_by_station,
        "raw_metrics": rows, "lead_metrics": lead_summary,
        "paired_key_rule": "exact station/init/target/lead/GHI-row keys required across A and AC and across seeds",
        "test_payload_opened": False,
        "claim_rule": ("validation selection is descriptive; OOF removes training-date overlap in upstream SimVP forecast generation, but independent test was not accessed"
                       if train_oof else
                       "in-sample upstream training forecasts make this screening only; do not claim forecast-input generalization"),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = a.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: summary[k] for k in ("state", "pairwise_delta_rmse_wm2",
        "pairwise_relative_rmse_change_pct", "date_block_bootstrap_95pct_ci_wm2",
        "target_bjt_date_blocks", "test_payload_opened")}, indent=2))


if __name__ == "__main__":
    main()
