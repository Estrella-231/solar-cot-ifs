#!/usr/bin/env python3
"""Validation-only paired GHI contrasts by keyed CPP centre COT and CLP."""
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
BINS = (
    ("CPP_nearzero_0_0.1", 0.0, 0.1),
    ("CPP_thin_0.1_5", 0.1, 5.0),
    ("CPP_moderate_5_30", 5.0, 30.0),
    ("CPP_thick_30_100", 30.0, 100.0000001),
    ("CPP_over_100", 100.0000001, float("inf")),
)


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_blocks(path):
    rows = load_rows(path)
    blocks = []
    for r in rows:
        blocks.append([x for x in r["bjt_dates"].split(";") if x])
    date_to_block = {d: i for i, block in enumerate(blocks) for d in block}
    return blocks, date_to_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cpp-cot", type=Path, required=True)
    ap.add_argument("--clp-reference", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    run = args.run
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    score = json.loads((run / "score_summary_fp64.json").read_text(encoding="utf-8"))
    if completion.get("state") != "COMPLETE_24_PAIRED_RUNS" or completion.get("test_payloads_read") != 0 or score.get("test_payloads_read") != 0:
        raise RuntimeError("training/score completion gate failed")
    rows = load_rows(run / "validation_rows.csv")
    inputs = np.load(run / "validation_inputs.npz", allow_pickle=False)
    pack_ids = np.asarray(inputs["pack_index"], dtype=np.int64)
    if len(rows) != len(pack_ids):
        raise RuntimeError(f"unexpected validation rows/ids: {len(rows)}/{len(pack_ids)}")
    if len(rows) != 4226 or len(pack_ids) != 4226:
        raise RuntimeError(f"expected 4,226 rows/ids, got {len(rows)}/{len(pack_ids)}")
    if len(set(pack_ids.tolist())) != len(pack_ids):
        raise RuntimeError("duplicate pack index in saved validation inputs")
    for i, row in enumerate(rows):
        row["pack_index"] = int(pack_ids[i])
        row["target_date_bjt"] = (datetime.fromisoformat(row["target_time_utc"]) + timedelta(hours=8)).date().isoformat()
    station_counts = {s: sum(r["station"] == s for r in rows) for s in STATIONS}
    if station_counts != {"sili": 1999, "zhujia": 2227}:
        raise RuntimeError(f"station support changed: {station_counts}")

    cpp = np.load(args.cpp_cot, allow_pickle=False)
    clp = np.load(args.clp_reference, allow_pickle=False)
    if not np.array_equal(cpp["sample_indices"], clp["sample_indices"]):
        raise RuntimeError("CPP COT and CLP reference sample indexes differ")
    if cpp["reference_cot"].shape != cpp["mask"].shape or clp["clp"].shape != clp["clp_valid"].shape:
        raise RuntimeError("CPP/CLP archive shapes mismatch")
    if cpp["reference_cot"].shape[1:] != (1, 16, 16) or clp["clp"].shape[1:] != (1, 16, 16):
        raise RuntimeError("unexpected reference patch geometry")
    ref_lut = {}
    for j, (station, stamp) in enumerate(zip(clp["station"], clp["target_time_utc"])):
        key = (str(station), str(stamp))
        if key in ref_lut:
            raise RuntimeError(f"duplicate CPP/CLP key {key}")
        ref_lut[key] = j
    cpp_center = np.full(len(rows), np.nan, dtype=np.float64)
    clp_center = np.full(len(rows), -1, dtype=np.int16)
    for i, row in enumerate(rows):
        j = ref_lut.get((row["station"], row["target_time_utc"]))
        if j is None:
            continue
        valid_cpp = bool(cpp["mask"][j, 0, 8, 8])
        valid_clp = bool(clp["clp_valid"][j, 0, 8, 8])
        common = bool(clp["common_mask"][j, 0, 8, 8])
        if valid_cpp and valid_clp and common:
            v = float(cpp["reference_cot"][j, 0, 8, 8])
            c = int(clp["clp"][j, 0, 8, 8])
            if np.isfinite(v) and 0 <= v <= 200 and c in (0, 1, 2):
                cpp_center[i] = v
                clp_center[i] = c

    # Load and enforce the exact common saved predictions for all 24 models.
    predictions = {}
    by_arm_station_seed = {}
    for seed in SEEDS:
        for station in STATIONS:
            wanted_ids = np.asarray(sorted(r["pack_index"] for r in rows if r["station"] == station), dtype=np.int64)
            for arm in ARMS:
                z = np.load(run / f"seed{seed}" / station / arm / "validation_predictions.npz", allow_pickle=False)
                ids = np.asarray(z["pack_index"], dtype=np.int64)
                pred = np.asarray(z["prediction"], dtype=np.float64)
                truth = np.asarray(z["truth"], dtype=np.float64)
                if not np.array_equal(ids, wanted_ids):
                    raise RuntimeError(f"prediction keys differ from validation rows: {seed}/{station}/{arm}")
                if not np.isfinite(pred).all() or not np.isfinite(truth).all():
                    raise RuntimeError(f"nonfinite saved predictions: {seed}/{station}/{arm}")
                predictions[(seed, station, arm)] = (ids, pred, truth)

    blocks, date_to_block = read_blocks(run / "date_blocks.csv")
    if len(blocks) != score.get("date_block_count"):
        raise RuntimeError("score and block manifest disagree")
    row_by_id = {r["pack_index"]: r for r in rows}
    row_date_by_id = {i: row_by_id[i]["target_date_bjt"] for i in row_by_id}
    group_masks = [("all_validation", np.ones(len(rows), dtype=bool))]
    cpp_valid = np.isfinite(cpp_center)
    group_masks.append(("CPP_supported", cpp_valid))
    for name, lo, hi in BINS:
        group_masks.append((name, cpp_valid & (cpp_center >= lo) & (cpp_center < hi)))
    group_masks.extend([
        ("CLP_clear_0", clp_center == 0),
        ("CLP_water_1", clp_center == 1),
        ("CLP_ice_2", clp_center == 2),
        ("CLP_cloud_1_or_2", np.isin(clp_center, [1, 2])),
    ])

    block_index_for_row = np.asarray([date_to_block[r["target_date_bjt"]] for r in rows], dtype=np.int64)
    row_station = np.asarray([r["station"] for r in rows])
    row_ids = np.asarray([r["pack_index"] for r in rows], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    out_groups = []
    for name, mask in group_masks:
        if not mask.any():
            continue
        selected_ids = row_ids[mask]
        per_run = {}
        block_sse = {}
        block_n = {}
        for seed in SEEDS:
            for station in STATIONS:
                use_mask = mask & (row_station == station)
                chosen_ids = row_ids[use_mask]
                if len(chosen_ids) == 0:
                    raise RuntimeError(f"empty station in stratum {name}: {station}")
                chosen_blocks = block_index_for_row[use_mask]
                for arm in ARMS:
                    ids, pred, truth = predictions[(seed, station, arm)]
                    take = np.isin(ids, chosen_ids, assume_unique=True)
                    if int(take.sum()) != len(chosen_ids):
                        raise RuntimeError(f"stratum prediction support mismatch: {name}/{seed}/{station}/{arm}")
                    resid = pred[take] - truth[take]
                    idx = np.searchsorted(ids, chosen_ids)
                    if not np.array_equal(ids[idx], chosen_ids):
                        raise RuntimeError("pack index lookup failure")
                    # `take` follows sorted ids, which are the same order as chosen_ids.
                    rmse = float(np.sqrt(np.mean(resid * resid, dtype=np.float64)))
                    mae = float(np.mean(np.abs(resid), dtype=np.float64))
                    mbe = float(np.mean(resid, dtype=np.float64))
                    per_run[(seed, station, arm)] = {"rmse": rmse, "mae": mae, "mbe": mbe, "n": len(resid)}
                    sse = np.bincount(chosen_blocks, weights=resid * resid, minlength=len(blocks)).astype(np.float64)
                    nn = np.bincount(chosen_blocks, minlength=len(blocks)).astype(np.float64)
                    block_sse[(seed, station, arm)] = sse
                    block_n[(seed, station, arm)] = nn

        # Draw from the full frozen validation block list, as in the primary
        # scorer. Reject only replicates with zero support for a station/seed
        # in this particular stratum; do not silently drop one-station blocks.
        station_block_support = []
        for station in STATIONS:
            support = np.zeros(len(blocks), dtype=bool)
            for seed in SEEDS:
                support |= block_n[(seed, station, ARMS[0])] > 0
            station_block_support.append(support)
        eligible_blocks = np.flatnonzero(np.logical_and.reduce(station_block_support))
        if len(eligible_blocks) < 2:
            raise RuntimeError(f"fewer than two common two-day date blocks for stratum {name}")
        accepted = []
        accepted_n = 0
        attempts = 0
        while accepted_n < args.bootstrap:
            batch_n = args.bootstrap - accepted_n
            candidate = rng.integers(0, len(blocks), size=(batch_n, len(blocks)), endpoint=False)
            valid_draw = np.ones(batch_n, dtype=bool)
            for seed in SEEDS:
                for station in STATIONS:
                    nn = block_n[(seed, station, ARMS[0])]
                    valid_draw &= nn[candidate].sum(axis=1) > 0
            if valid_draw.any():
                kept = candidate[valid_draw][:args.bootstrap - accepted_n]
                accepted.append(kept)
                accepted_n += len(kept)
            attempts += batch_n
            if attempts > args.bootstrap * 1000:
                raise RuntimeError(f"too few valid bootstrap draws for stratum {name}")
        draw_blocks = np.concatenate(accepted, axis=0)[:args.bootstrap]
        boot_rmse = {}
        for arm in ARMS:
            runs_rmse = []
            for seed in SEEDS:
                for station in STATIONS:
                    sse = block_sse[(seed, station, arm)]
                    nn = block_n[(seed, station, arm)]
                    bs_sse = sse[draw_blocks].sum(axis=1, dtype=np.float64)
                    bs_n = nn[draw_blocks].sum(axis=1, dtype=np.float64)
                    if np.any(bs_n <= 0):
                        raise RuntimeError(f"zero support in eligible-block bootstrap: {name}/{seed}/{station}/{arm}")
                    runs_rmse.append(np.sqrt(bs_sse / bs_n))
            boot_rmse[arm] = np.mean(np.stack(runs_rmse, axis=0), axis=0)
        equal_rmse = {arm: float(np.mean([per_run[(seed, station, arm)]["rmse"] for seed in SEEDS for station in STATIONS])) for arm in ARMS}
        station_mean = {station: {arm: float(np.mean([per_run[(seed, station, arm)]["rmse"] for seed in SEEDS])) for arm in ARMS} for station in STATIONS}
        seed_mean = {str(seed): {arm: float(np.mean([per_run[(seed, station, arm)]["rmse"] for station in STATIONS])) for arm in ARMS} for seed in SEEDS}
        contrasts = {}
        for cand, base in (("M11", "M10"), ("M11", "M01"), ("M10", "M00"), ("M01", "M00"), ("M11", "M00")):
            delta = boot_rmse[cand] - boot_rmse[base]
            ci = np.percentile(delta, [2.5, 97.5])
            relative = 100.0 * (boot_rmse[base] - boot_rmse[cand]) / boot_rmse[base]
            pci = np.percentile(relative, [2.5, 97.5])
            contrasts[f"{cand}_minus_{base}"] = {
                "RMSE_delta_Wm2": equal_rmse[cand] - equal_rmse[base],
                "relative_reduction_percent": 100.0 * (equal_rmse[base] - equal_rmse[cand]) / equal_rmse[base],
                "date_block_CI_delta_Wm2": [float(ci[0]), float(ci[1])],
                "date_block_CI_relative_reduction_percent": [float(pci[0]), float(pci[1])],
                "seed_equal_station_delta_Wm2": {str(seed): float(seed_mean[str(seed)][cand] - seed_mean[str(seed)][base]) for seed in SEEDS},
                "station_mean_delta_Wm2": {station: float(station_mean[station][cand] - station_mean[station][base]) for station in STATIONS},
            }
        out_groups.append({
            "group": name, "n": int(mask.sum()),
            "station_n": {s: int(np.count_nonzero(mask & (row_station == s))) for s in STATIONS},
            "bjt_dates": int(len({r["target_date_bjt"] for r, keep in zip(rows, mask) if keep})),
            "common_two_day_blocks": int(len(eligible_blocks)),
            "common_block_dates": [[d for d in blocks[int(b)]] for b in eligible_blocks],
            "bootstrap_block_count": int(len(blocks)),
            "bootstrap_valid_draws": int(args.bootstrap),
            "bootstrap_draws_rejected_for_empty_station_support": int(attempts - accepted_n),
            "equal_station_equal_seed_RMSE_Wm2": equal_rmse,
            "station_RMSE_Wm2": station_mean,
            "seed_equal_station_RMSE_Wm2": seed_mean,
            "contrasts": contrasts,
            "interval_scope": "pointwise conditional validation date-block bootstrap; exploratory; no multiplicity adjustment",
        })

    summary = {
        "state": "COMPLETE_VALIDATION_CPP_CLP_STRATA",
        "split_id": json.loads((run / "input_source_manifest.json").read_text(encoding="utf-8"))["split_id"],
        "evaluation_scope": "same-time station GHI validation; exploratory, not independent test",
        "validation_rows": len(rows), "CPP_COT_CLP_matched_rows": int(cpp_valid.sum()),
        "CPP_COT_missing_or_invalid_rows": int((~cpp_valid).sum()),
        "CPP_reference_semantics": "co-produced retrieval reference; stratification only, not independent truth",
        "CPP_COT_definition": "physical CPP reference COT at station-centre pixel [8,8] of the station 16x16 crop; no patch mean",
        "CLP_semantics": "CPP co-produced class 0 clear, 1 water cloud, 2 ice cloud; centre pixel, not independent truth or official CLM",
        "bins": {name: [lo, hi if np.isfinite(hi) else None] for name, lo, hi in BINS},
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed,
            "unit": "shared two-day BJT validation blocks across the full validation block list; replicates with zero stratum support for any station/seed are rejected",
                      "interval": "pointwise percentile 95%, no multiplicity adjustment"},
        "test_payloads_read": 0,
        "sources": {"CPP_reference_archive": str(args.cpp_cot), "CLP_key_archive": str(args.clp_reference)},
        "groups": out_groups,
    }
    write_json(run / "strata_summary_fp64.json", summary)
    cols = ["group", "n", "sili_n", "zhujia_n", "common_two_day_blocks", "M00_RMSE", "M10_RMSE", "M01_RMSE", "M11_RMSE", "M11_minus_M10_Wm2", "M11_minus_M10_CI_low", "M11_minus_M10_CI_high", "M11_minus_M01_Wm2", "M11_minus_M01_CI_low", "M11_minus_M01_CI_high"]
    with (run / "strata_summary_fp64.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in out_groups:
            c1 = g["contrasts"]["M11_minus_M10"]
            c2 = g["contrasts"]["M11_minus_M01"]
            w.writerow({"group": g["group"], "n": g["n"], "sili_n": g["station_n"]["sili"], "zhujia_n": g["station_n"]["zhujia"],
                        "common_two_day_blocks": g["common_two_day_blocks"],
                        "M00_RMSE": g["equal_station_equal_seed_RMSE_Wm2"]["M00"], "M10_RMSE": g["equal_station_equal_seed_RMSE_Wm2"]["M10"],
                        "M01_RMSE": g["equal_station_equal_seed_RMSE_Wm2"]["M01"], "M11_RMSE": g["equal_station_equal_seed_RMSE_Wm2"]["M11"],
                        "M11_minus_M10_Wm2": c1["RMSE_delta_Wm2"], "M11_minus_M10_CI_low": c1["date_block_CI_delta_Wm2"][0], "M11_minus_M10_CI_high": c1["date_block_CI_delta_Wm2"][1],
                        "M11_minus_M01_Wm2": c2["RMSE_delta_Wm2"], "M11_minus_M01_CI_low": c2["date_block_CI_delta_Wm2"][0], "M11_minus_M01_CI_high": c2["date_block_CI_delta_Wm2"][1]})
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
