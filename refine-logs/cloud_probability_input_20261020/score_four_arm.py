#!/usr/bin/env python3
"""Independent FP64 scoring and shared two-day date-block bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ARMS = ("M00", "M10", "M01", "M11")
STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def make_date_blocks(rows):
    dates = sorted({date.fromisoformat(r["target_date_bjt"]) for r in rows})
    by_month = {}
    for d in dates:
        by_month.setdefault((d.year, d.month), []).append(d)
    blocks = []
    for month in sorted(by_month):
        mdays = by_month[month]
        runs = []
        cur = []
        for d in mdays:
            if cur and d != cur[-1] + timedelta(days=1):
                runs.append(cur)
                cur = []
            cur.append(d)
        if cur:
            runs.append(cur)
        for run in runs:
            for i in range(0, len(run), 2):
                blocks.append(run[i:i + 2])
    if not blocks or any(len(b) > 2 for b in blocks):
        raise RuntimeError("invalid two-day validation blocks")
    date_to_block = {d.isoformat(): i for i, b in enumerate(blocks) for d in b}
    return blocks, date_to_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    completion_path = args.run / "completion.json"
    if not completion_path.exists():
        raise RuntimeError("training completion receipt is absent; partial results are not scored")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    setup = json.loads((args.run / "setup.json").read_text(encoding="utf-8"))
    if completion.get("state") != "COMPLETE_24_PAIRED_RUNS" or completion.get("test_payloads_read") != 0 or setup.get("test_payloads_read") != 0:
        raise RuntimeError("completion/test gate failed")
    rows = read_csv(args.run / "validation_rows.csv")
    if len(rows) != 4226:
        raise RuntimeError(f"expected 4,226 cleaned validation rows, got {len(rows)}")
    observed_station_counts = {s: sum(r["station"] == s for r in rows) for s in STATIONS}
    if observed_station_counts != {"sili": 1999, "zhujia": 2227}:
        raise RuntimeError(f"cleaned validation station counts differ from the frozen contract: {observed_station_counts}")
    if any(r["station"] == "sili" and (datetime.fromisoformat(r["target_time_utc"]) + timedelta(hours=8)).date().isoformat() == "2024-09-05" for r in rows):
        raise RuntimeError("excluded Sili station-day leaked into scored validation rows")
    source_rows = read_csv(args.pack / "rows.csv")
    source_index = {(r["station"], r["target_time_utc"]): i for i, r in enumerate(source_rows)}
    if len(source_index) != len(source_rows):
        raise RuntimeError("source pack has duplicate station-target rows")
    row_by_index = {}
    for r in rows:
        i = source_index.get((r["station"], r["target_time_utc"]))
        if i is None:
            raise RuntimeError(f"validation output key absent from source pack: {r['station']}/{r['target_time_utc']}")
        if source_rows[i]["split"] != "val":
            raise RuntimeError(f"validation output contains a non-validation source row: {r['station']}/{r['target_time_utc']}")
        dt = datetime.fromisoformat(r["target_time_utc"]) + timedelta(hours=8)
        r["target_date_bjt"] = dt.date().isoformat()
        if i in row_by_index:
            raise RuntimeError(f"duplicate validation source pack index {i}")
        row_by_index[i] = r
    blocks, date_to_block = make_date_blocks(rows)
    with (args.run / "date_blocks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_index", "bjt_dates", "n_station_rows"])
        for bi, block in enumerate(blocks):
            dstrings = [d.isoformat() for d in block]
            nrows = sum(r["target_date_bjt"] in dstrings for r in rows)
            w.writerow([bi, ";".join(dstrings), nrows])

    predictions = {}
    run_rows = []
    common_by_station = {}
    for seed in SEEDS:
        for station in STATIONS:
            for arm in ARMS:
                folder = args.run / f"seed{seed}" / station / arm
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                z = np.load(folder / "validation_predictions.npz", allow_pickle=False)
                ids = np.asarray(z["pack_index"], dtype=np.int64)
                pred = np.asarray(z["prediction"], dtype=np.float64)
                cpu = np.asarray(z["cpu_prediction"], dtype=np.float64)
                truth = np.asarray(z["truth"], dtype=np.float64)
                if len(set(ids.tolist())) != len(ids) or len(ids) != len(pred) or len(ids) != len(truth):
                    raise RuntimeError(f"malformed predictions: {seed}/{station}/{arm}")
                if np.max(np.abs(pred - cpu)) > 0.1:
                    raise RuntimeError(f"CPU reload mismatch: {seed}/{station}/{arm}")
                if station not in common_by_station:
                    common_by_station[station] = (ids.copy(), truth.copy())
                elif not np.array_equal(ids, common_by_station[station][0]) or not np.array_equal(truth, common_by_station[station][1]):
                    raise RuntimeError(f"paired arms/seeds do not share identical rows and labels: {seed}/{station}/{arm}")
                expected_station = [i for i, r in row_by_index.items() if r["station"] == station]
                if ids.tolist() != sorted(expected_station):
                    raise RuntimeError(f"prediction row mapping differs from cleaned validation rows: {seed}/{station}/{arm}")
                # The training script casts labels to float32 before placing
                # them on GPU; mirror that explicit source conversion before
                # the independent FP64 metric calculation.
                expected_truth = np.asarray([float(np.float32(float(row_by_index[int(i)]["ghi"]))) for i in ids], dtype=np.float64)
                if not np.array_equal(truth, expected_truth):
                    raise RuntimeError(f"saved truth differs from source pack: {seed}/{station}/{arm}")
                residual = pred - truth
                rmse = float(np.sqrt(np.mean(residual * residual, dtype=np.float64)))
                mae = float(np.mean(np.abs(residual), dtype=np.float64))
                mbe = float(np.mean(residual, dtype=np.float64))
                if abs(rmse - float(metrics["rmse_Wm2"])) > 1e-7:
                    raise RuntimeError(f"saved arm metric does not reproduce in FP64: {seed}/{station}/{arm}")
                predictions[(seed, station, arm)] = (ids, pred, truth)
                run_rows.append({"seed": seed, "station": station, "arm": arm, "n": len(ids),
                                 "rmse_Wm2_FP64": rmse, "mae_Wm2_FP64": mae,
                                 "mbe_Wm2_FP64": mbe, "best_epoch": metrics["best_epoch"],
                                 "updates": metrics["updates"], "independent_recompute": "PASS"})

    with (args.run / "per_run_metrics_fp64.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)

    # Index each run by validation BJT date and date-block cluster. The same
    # block draws are applied to all arms, stations and seeds.
    pair_index = {}
    for key, (ids, pred, truth) in predictions.items():
        rows_for_run = [row_by_index[int(i)] for i in ids]
        bidx = np.asarray([date_to_block[r["target_date_bjt"]] for r in rows_for_run], dtype=np.int64)
        sse = np.bincount(bidx, weights=(pred - truth) ** 2, minlength=len(blocks)).astype(np.float64)
        nobs = np.bincount(bidx, minlength=len(blocks)).astype(np.float64)
        pair_index[key] = (sse, nobs)

    point_rmse = {}
    for arm in ARMS:
        vals = [r["rmse_Wm2_FP64"] for r in run_rows if r["arm"] == arm]
        point_rmse[arm] = float(np.mean(vals))

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(blocks), size=(args.bootstrap, len(blocks)), endpoint=False)
    bootstrap_rmse = {}
    for arm in ARMS:
        per_pair = []
        for seed in SEEDS:
            for station in STATIONS:
                sse, nobs = pair_index[(seed, station, arm)]
                bs_sse = sse[draws].sum(axis=1, dtype=np.float64)
                bs_n = nobs[draws].sum(axis=1, dtype=np.float64)
                if np.any(bs_n <= 0):
                    raise RuntimeError("bootstrap draw has no validation observations for a station/seed")
                per_pair.append(np.sqrt(bs_sse / bs_n))
        bootstrap_rmse[arm] = np.mean(np.stack(per_pair, axis=0), axis=0)

    contrasts = []
    for candidate, baseline, name in (
        ("M11", "M10", "COT_plus_cloud_vs_COT"),
        ("M11", "M01", "COT_plus_cloud_vs_cloud_only"),
        ("M10", "M00", "COT_only_vs_base"),
        ("M01", "M00", "cloud_only_vs_base"),
        ("M11", "M00", "joint_vs_base"),
    ):
        delta = bootstrap_rmse[candidate] - bootstrap_rmse[baseline]
        q = np.percentile(delta, [2.5, 97.5])
        point_delta = point_rmse[candidate] - point_rmse[baseline]
        point_pct = 100.0 * (point_rmse[baseline] - point_rmse[candidate]) / point_rmse[baseline]
        base_draws = bootstrap_rmse[baseline]
        pct_draws = 100.0 * (base_draws - bootstrap_rmse[candidate]) / base_draws
        pct_ci = np.percentile(pct_draws, [2.5, 97.5])
        seed_delta = {}
        for seed in SEEDS:
            c = [r["rmse_Wm2_FP64"] for r in run_rows if r["seed"] == seed and r["arm"] == candidate]
            b = [r["rmse_Wm2_FP64"] for r in run_rows if r["seed"] == seed and r["arm"] == baseline]
            seed_delta[str(seed)] = float(np.mean(c) - np.mean(b))
        station_delta = {}
        for station in STATIONS:
            c = [r["rmse_Wm2_FP64"] for r in run_rows if r["station"] == station and r["arm"] == candidate]
            b = [r["rmse_Wm2_FP64"] for r in run_rows if r["station"] == station and r["arm"] == baseline]
            station_delta[station] = float(np.mean(c) - np.mean(b))
        contrasts.append({
            "name": name, "candidate": candidate, "baseline": baseline,
            "candidate_minus_baseline_RMSE_Wm2": point_delta,
            "relative_RMSE_reduction_percent": point_pct,
            "date_block_95pct_CI_delta_Wm2": [float(q[0]), float(q[1])],
            "date_block_95pct_CI_relative_reduction_percent": [float(pct_ci[0]), float(pct_ci[1])],
            "seed_equal_station_delta_Wm2": seed_delta,
            "station_mean_delta_Wm2": station_delta,
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_block_count": len(blocks),
        })

    c1 = next(x for x in contrasts if x["name"] == "COT_plus_cloud_vs_COT")
    c2 = next(x for x in contrasts if x["name"] == "COT_plus_cloud_vs_cloud_only")
    c1_criterion = {
        "three_seed_mean_delta_negative": c1["candidate_minus_baseline_RMSE_Wm2"] < 0,
        "at_least_two_of_three_seed_deltas_negative": sum(v < 0 for v in c1["seed_equal_station_delta_Wm2"].values()) >= 2,
        "both_station_mean_deltas_nonpositive": all(v <= 0 for v in c1["station_mean_delta_Wm2"].values()),
        "date_block_CI_upper_below_zero": c1["date_block_95pct_CI_delta_Wm2"][1] < 0,
    }
    c1_criterion["C1_supported_by_frozen_rule"] = all(c1_criterion.values())
    c2_criterion = {
        "three_seed_mean_delta_negative": c2["candidate_minus_baseline_RMSE_Wm2"] < 0,
        "at_least_two_of_three_seed_deltas_negative": sum(v < 0 for v in c2["seed_equal_station_delta_Wm2"].values()) >= 2,
        "both_station_mean_deltas_nonpositive": all(v <= 0 for v in c2["station_mean_delta_Wm2"].values()),
        "date_block_CI_upper_below_zero": c2["date_block_95pct_CI_delta_Wm2"][1] < 0,
    }
    c2_criterion["C2_supported_by_frozen_rule"] = all(c2_criterion.values())
    summary = {
        "state": "COMPLETE_FP64_PAIRED_SCORE",
        "evaluation_scope": "frozen monthly validation; exploratory, not independent test",
        "sample_count": len(rows), "station_count": 2, "seed_count": 3,
        "equal_station_equal_seed_mean_RMSE_Wm2": point_rmse,
        "contrasts": contrasts,
        "frozen_decision_rule": {"C1": c1_criterion, "C2": c2_criterion,
                                  "C2_claim_status": "descriptive_only_until_C1_passes" if not c1_criterion["C1_supported_by_frozen_rule"] else "evaluate_per_plan"},
        "date_block_definition": "within each month, maximal runs of consecutive BJT validation dates, split into consecutive chunks of two; final singleton retained if required",
        "date_block_count": len(blocks),
        "date_blocks": [[d.isoformat() for d in b] for b in blocks],
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
        "bootstrap_unit": "shared validation BJT date blocks across all arms, stations and seeds",
        "scoring_dtype": "float64 independently recomputed from saved per-row predictions and labels",
        "test_payloads_read": 0,
    }
    write_json(args.run / "score_summary_fp64.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
