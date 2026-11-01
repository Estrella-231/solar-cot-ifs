#!/usr/bin/env python3
"""Independent paired scoring and date-block uncertainty for residual runs."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


ARMS = ("A", "AGRI", "AGRI_COT")
SEEDS = (42, 43, 44)
STATIONS = ("sili", "zhujia")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric(pred, truth):
    err = np.asarray(pred, np.float64) - np.asarray(truth, np.float64)
    return {"rmse_wm2": float(np.sqrt(np.mean(err * err))),
            "mae_wm2": float(np.mean(np.abs(err))),
            "bias_wm2": float(np.mean(err))}


def bjt_day(utc_text):
    utc = datetime.strptime(utc_text, "%Y-%m-%d %H:%M:%S")
    return (utc - timedelta(hours=8)).date().isoformat()


def find_run(root, station, seed):
    if station == "sili":
        return root / "signed_residual_direction_20261014_v3" / f"seed{seed}" / station
    return root / "signed_residual_zhujia_20261014_v1" / f"seed{seed}" / station


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261014)
    args = p.parse_args()

    pack_rows = read_csv(args.pack / "rows.csv")
    pack_index = {(r["station"], r["target_time_utc"]): i
                  for i, r in enumerate(pack_rows)}
    cot = np.load(args.pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    runs = {}
    point_metrics = []
    direction_metrics = []
    val_dates = set()
    for station in STATIONS:
        for seed in SEEDS:
            d = find_run(args.results_root, station, seed)
            keys = read_csv(d / "validation_keys.csv")
            a = np.load(d / "A_reloaded_val_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)
            preds = {"A": a}
            deltas = {}
            for arm in ("AGRI", "AGRI_COT"):
                preds[arm] = np.load(d / f"{arm}_val_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)
                deltas[arm] = np.load(d / f"{arm}_val_signed_residual_wm2.npy", allow_pickle=False).astype(np.float64)
            if any(len(v) != len(keys) for v in [*preds.values(), *deltas.values()]):
                raise ValueError(f"prediction/key length mismatch: {station}/{seed}")
            if any(r["station"] != station or r["split"] != "val" for r in keys):
                raise ValueError(f"wrong validation key cohort: {station}/{seed}")
            key_ids = [(r["station"], r["target_time_utc"]) for r in keys]
            if len(set(key_ids)) != len(key_ids):
                raise ValueError(f"duplicate validation keys: {station}/{seed}")
            if key_ids not in [v["key_ids"] for v in runs.values() if v["station"] == station]:
                prior = [v for v in runs.values() if v["station"] == station]
                if prior and key_ids != prior[0]["key_ids"]:
                    raise ValueError(f"paired keys differ by seed: {station}/{seed}")
            truth = np.asarray([float(r["ghi"]) for r in keys], np.float64)
            dates = np.asarray([bjt_day(r["target_time_utc"]) for r in keys])
            val_dates.update(dates.tolist())
            cots = np.asarray([np.expm1(float(cot[pack_index[k], 0, 1, 1]))
                               for k in key_ids], np.float64)
            clear = np.asarray([float(r["clear"]) for r in keys], np.float64)
            observed_kt = np.asarray([float(r["kt"]) for r in keys], np.float64)
            score = {arm: metric(preds[arm], truth) for arm in ARMS}
            metadata = json.loads((d / "AGRI_validation.json").read_text())
            metadata_cot = json.loads((d / "AGRI_COT_validation.json").read_text())
            if metadata["trainable_parameters"] != metadata_cot["trainable_parameters"]:
                raise ValueError("capacity mismatch in saved run metadata")
            if metadata["step_zero_identity_max_abs_wm2"] != 0.0 or metadata_cot["step_zero_identity_max_abs_wm2"] != 0.0:
                raise ValueError("step-zero identity gate failed")
            saved_a_delta = json.loads((d / "A_reload_check.json").read_text())["saved_A_max_abs_delta_wm2"]
            if saved_a_delta > 1.0:
                raise ValueError("A reload numerical difference exceeds accepted tolerance")
            run_key = (station, seed)
            runs[run_key] = {"station": station, "seed": seed, "key_ids": key_ids,
                             "keys": keys, "truth": truth, "dates": dates,
                             "preds": preds, "deltas": deltas, "score": score,
                             "cots": cots, "clear": clear, "observed_kt": observed_kt}
            rec = {"station": station, "seed": seed, "n": len(keys),
                   "A_reload_max_abs_delta_wm2": saved_a_delta,
                   "trainable_parameters_each_residual_arm": metadata["trainable_parameters"]}
            for arm in ARMS:
                for k, v in score[arm].items():
                    rec[f"{arm}_{k}"] = v
            rec["AGRI_minus_A_RMSE_wm2"] = score["AGRI"]["rmse_wm2"] - score["A"]["rmse_wm2"]
            rec["AGRI_COT_minus_A_RMSE_wm2"] = score["AGRI_COT"]["rmse_wm2"] - score["A"]["rmse_wm2"]
            rec["AGRI_COT_minus_AGRI_RMSE_wm2"] = score["AGRI_COT"]["rmse_wm2"] - score["AGRI"]["rmse_wm2"]
            point_metrics.append(rec)

            under = truth > a
            severe = (truth - a) >= 100.0
            high_bright = (cots >= 3.0) & (clear >= 100.0) & (observed_kt >= 0.9)
            for arm in ("AGRI", "AGRI_COT"):
                delta = deltas[arm]
                corrected = preds[arm]
                for label, mask in (("A_underpredict", under),
                                    ("A_underpredict_by_100plus", severe),
                                    ("COT_ge3_clear_ge100_observed_kt_ge0p9", high_bright)):
                    if not np.any(mask):
                        continue
                    direction_metrics.append({
                        "station": station, "seed": seed, "arm": arm,
                        "subset": label, "n": int(mask.sum()),
                        "positive_adjustment_fraction": float(np.mean(delta[mask] > 0)),
                        "right_sign_fraction": float(np.mean(delta[mask] * (truth[mask] - a[mask]) > 0)),
                        "absolute_error_improved_fraction": float(np.mean(np.abs(corrected[mask] - truth[mask]) < np.abs(a[mask] - truth[mask]))),
                        "A_rmse_wm2": metric(a[mask], truth[mask])["rmse_wm2"],
                        "candidate_rmse_wm2": metric(corrected[mask], truth[mask])["rmse_wm2"],
                    })

    if len(runs) != len(STATIONS) * len(SEEDS):
        raise ValueError("expected six station/seed pairs")
    dates = sorted(val_dates)
    date_index = {d: i for i, d in enumerate(dates)}
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(dates), size=(args.bootstrap, len(dates)))
    boot_out = {}
    for left, right, label in (("AGRI", "A", "AGRI_residual_minus_A"),
                               ("AGRI_COT", "A", "AGRI_COT_residual_minus_A"),
                               ("AGRI_COT", "AGRI", "COT_increment_vs_AGRI_residual")):
        station_boot = {}
        for station in STATIONS:
            seed_rmse = {left: [], right: []}
            for arm in (left, right):
                rows_sse, rows_n = [], []
                for seed in SEEDS:
                    run = runs[(station, seed)]
                    di = np.asarray([date_index[d] for d in run["dates"]], np.int64)
                    n_day = np.bincount(di, minlength=len(dates)).astype(np.float64)
                    sse = np.vstack([np.bincount(
                        di, weights=(run["preds"][arm] - run["truth"]) ** 2,
                        minlength=len(dates))])
                    # Each seed shares the same station/date validation cohort.
                    rows_sse.append(sse[0]); rows_n.append(n_day)
                sse_by_seed = np.stack(rows_sse, axis=0)
                n_by_seed = np.stack(rows_n, axis=0)
                sampled_sse = sse_by_seed[:, draws].sum(axis=2)
                sampled_n = n_by_seed[:, draws].sum(axis=2)
                seed_rmse[arm] = np.sqrt(sampled_sse / sampled_n)
            station_boot[station] = seed_rmse[left].mean(axis=0) - seed_rmse[right].mean(axis=0)
        all_boot = np.mean(np.stack([station_boot[s] for s in STATIONS]), axis=0)
        point_by_station = {}
        for station in STATIONS:
            aleft = np.mean([runs[(station, sd)]["score"][left]["rmse_wm2"] for sd in SEEDS])
            aright = np.mean([runs[(station, sd)]["score"][right]["rmse_wm2"] for sd in SEEDS])
            point_by_station[station] = float(aleft - aright)
        point = float(np.mean(list(point_by_station.values())))
        boot_out[label] = {"left_minus_right_rmse_wm2": point,
                           "station_equal_point_by_station": point_by_station,
                           "date_block_bootstrap_95pct_ci_wm2": [float(x) for x in np.quantile(all_boot, [.025, .975])],
                           "bootstrap_replicates": args.bootstrap,
                           "bootstrap_unit": "BJT target-date blocks sampled in common across both stations; seeds are averaged within station",
                           "ci_interpretation": "validation-only and conditional on checkpoints already selected by validation RMSE"}

    station_equal = {}
    for arm in ARMS:
        by_station = {}
        for station in STATIONS:
            by_station[station] = float(np.mean([runs[(station, seed)]["score"][arm]["rmse_wm2"] for seed in SEEDS]))
        station_equal[arm] = {"rmse_by_station_mean_over_seeds": by_station,
                              "equal_station_mean_rmse_wm2": float(np.mean(list(by_station.values())))}

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_metrics[0].keys()))
        writer.writeheader(); writer.writerows(point_metrics)
    with (args.output / "signed_direction_subgroups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(direction_metrics[0].keys()))
        writer.writeheader(); writer.writerows(direction_metrics)
    summary = {"state": "INDEPENDENT_RECOMPUTATION_COMPLETE",
               "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
               "test_payload_opened": False, "station_seed_pairs": len(runs),
               "seeds": list(SEEDS),
               "training_receipts": {
                   "sili": {"pbs_job_id": "213950.tc6000", "state": "COMPLETE", "exit_status": 0,
                            "node": "node21", "gpu": "NVIDIA A100-SXM4-40GB"},
                   "zhujia": {"pbs_job_id": "213952.tc6000", "state": "COMPLETE", "exit_status": 0,
                               "node": "node21", "gpu": "NVIDIA A100-SXM4-40GB"}},
               "cudnn_enabled": False,
               "A_reload_tolerance_wm2": 1.0,
               "validation_dates_bjt": len(dates), "point_metrics": station_equal,
               "paired_date_block_uncertainty": boot_out,
               "direction_subgroup_rule": "diagnostic masks use observed labels and retrieved COT only after prediction; neither is an input gate",
               "baseline_A_reload_max_abs_wm2": max(
                   json.loads((find_run(args.results_root, s, sd) / "A_reload_check.json").read_text())["saved_A_max_abs_delta_wm2"]
                   for s in STATIONS for sd in SEEDS)}
    (args.output / "independent_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
