#!/usr/bin/env python3
"""FP64 paired scoring for the original-train-date outer holdout.

This is an internal station-center comparison against CPP retrieval reference,
not an independent truth or monthly-validation/test score.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = tuple(range(15, 241, 15))
SITES = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LOG_DENOM = math.log(101.0)
SEEDS = (42, 43, 44)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_prediction(path: Path) -> tuple[list[str], np.ndarray]:
    z = np.load(path, allow_pickle=False)
    times = [str(x) for x in z["init_time_bjt"].tolist()]
    pred = np.asarray(z["pred_log_scaled"], dtype=np.float64)
    expected = (len(times), 16, 1, 16, 16)
    if pred.shape != expected or len(set(times)) != len(times):
        raise ValueError(f"bad or duplicate prediction schema {path}: {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"non-finite saved prediction: {path}")
    return times, pred


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    if args.bootstrap < 100:
        raise ValueError("use at least 100 date-block bootstrap draws")

    build = json.loads((args.cache_root / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    audit = json.loads((args.cache_root / "CACHE_AUDIT.json").read_text(encoding="utf-8"))
    if build.get("state") != "COMPLETE_OOF_SOURCE_PAIRED_CACHE_BUILD":
        raise ValueError("cache build did not complete")
    if audit.get("state") != "PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED":
        raise ValueError("cache audit did not pass")
    for record in (build, audit):
        if any(record.get(k) != 0 for k in (
            "official_monthly_validation_payloads_read", "test_payloads_read", "buffer_payloads_read"
        )):
            raise ValueError("monthly validation/test/buffer must remain closed")

    prediction: dict[str, dict[str, np.ndarray]] = {}
    init_keys: dict[str, set[str]] = {}
    for seed in SEEDS:
        for arm in ("fullr_control", "oof_train_source"):
            name = f"{arm}_seed{seed}"
            run = args.run_root / name
            complete = json.loads((run / "training_complete.json").read_text(encoding="utf-8"))
            if complete.get("test_payloads_read") != 0 or complete.get("buffer_payloads_read") != 0:
                raise ValueError(f"test/buffer gate failed in {name}")
            if complete.get("state") != "COMPLETE_TRAINED_HUBEI_INTERNAL_TRAIN_DATE_HOLDOUT_COT_DIAGNOSTIC":
                raise ValueError(f"unexpected completion state in {name}")
            if (complete.get("seed") != seed or complete.get("global_step") != 1350
                    or complete.get("global_batch") != 32 or complete.get("world_size") != 1
                    or complete.get("train_config", {}).get("max_epochs") != 10):
                raise ValueError(f"training contract mismatch in {name}: {complete}")
            pred_path = run / "outer_train_holdout_predictions.npz"
            if complete.get("outer_holdout_predictions_sha256") != sha256(pred_path):
                raise ValueError(f"saved prediction hash mismatch in {name}")
            if complete.get("best_checkpoint_sha256") != sha256(run / "best.pt"):
                raise ValueError(f"best checkpoint hash mismatch in {name}")
            keys, values = read_prediction(pred_path)
            prediction[name] = {k: values[i] for i, k in enumerate(keys)}
            init_keys[name] = set(keys)
    common_init = set.intersection(*init_keys.values())
    if not common_init or any(v != common_init for v in init_keys.values()):
        raise ValueError("all six arms must save exactly the same outer-holdout init keys")

    bank_rows = read_csv(args.cache_root / "fullr_control" / "cache" / "bank_rows.csv")
    bank_index: dict[str, int] = {}
    split: dict[str, str] = {}
    for row in bank_rows:
        t = row["time_bjt"]
        if t in bank_index:
            raise ValueError(f"duplicate cache time {t}")
        bank_index[t] = int(row["index"])
        split[t] = row["split"].lower()

    cache_dir = args.cache_root / "fullr_control" / "cache"
    arrays = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("history_cot_log", "history_valid", "cpp_cot", "cpp_valid")
    }
    pairs = [r for r in read_csv(cache_dir / "forecast_pairs.csv") if r["split"].lower() == "val"]
    # Stats are accumulated in FP64 and grouped by target date for paired block bootstrap.
    # Tuple: squared error, absolute error, signed error, count.
    stats: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    date_set: set[str] = set()
    scored_pairs: set[tuple[str, int, str]] = set()
    pixel_records = 0
    stratum_pixel_records = {name: 0 for name in ("near_zero_lt1", "thin_1_10", "moderate_10_30", "thick_ge30")}
    stratum_date_support = {name: set() for name in stratum_pixel_records}
    out_of_range = {name: 0 for name in prediction}
    confusion = {name: [0, 0, 0] for name in [*prediction, "persistence"]}

    for row in pairs:
        init_s = row["init_time_bjt"]
        target_s = row["target_time_bjt"]
        lead = int(row["lead_minutes"])
        init_t = dt.datetime.strptime(init_s, "%Y%m%d%H%M")
        target_t = dt.datetime.strptime(target_s, "%Y%m%d%H%M")
        if init_s not in common_init:
            continue
        if target_t - init_t != dt.timedelta(minutes=lead) or lead not in LEADS:
            raise ValueError(f"invalid outer key/lead: {row}")
        if split.get(init_s) != "val" or split.get(target_s) != "val":
            raise ValueError(f"cross-partition pair in outer holdout: {row}")
        init_i = bank_index[init_s]
        target_i = bank_index[target_s]
        lead_i = lead // 15 - 1
        date = target_t.strftime("%Y%m%d")
        date_set.add(date)
        history_log = np.asarray(arrays["history_cot_log"][init_i, 0], dtype=np.float64)
        history_valid = np.asarray(arrays["history_valid"][init_i, 0], dtype=bool)
        persistence = np.expm1(history_log * LOG_DENOM)
        truth_map = np.asarray(arrays["cpp_cot"][target_i, 0], dtype=np.float64)
        truth_valid = np.asarray(arrays["cpp_valid"][target_i, 0], dtype=bool)

        for site, (rr, cc) in SITES.items():
            if not (truth_valid[rr, cc] and history_valid[rr, cc]):
                continue
            truth = float(truth_map[rr, cc])
            if not math.isfinite(truth):
                raise ValueError(f"non-finite CPP reference at {target_s}/{site}")
            scored_pairs.add((init_s, lead, target_s))
            pixel_records += 1
            stratum = "near_zero_lt1" if truth < 1 else "thin_1_10" if truth < 10 else "moderate_10_30" if truth < 30 else "thick_ge30"
            stratum_pixel_records[stratum] += 1
            stratum_date_support[stratum].add(date)
            forecasts: dict[str, float] = {}
            for name, maps in prediction.items():
                estimate_map = np.expm1(maps[init_s][lead_i, 0] * LOG_DENOM)
                estimate = float(estimate_map[rr, cc])
                if not math.isfinite(estimate):
                    raise ValueError(f"non-finite COT prediction in {name}/{init_s}/{lead}")
                out_of_range[name] += int(estimate < 0 or estimate > 100)
                forecasts[name] = estimate
            estimates = {**forecasts, "persistence": float(persistence[rr, cc])}
            for name, estimate in estimates.items():
                error = estimate - truth
                for bucket in ("all", stratum):
                    acc = stats[(name, site, lead, date, bucket)]
                    acc[0] += error * error
                    acc[1] += abs(error)
                    acc[2] += error
                    acc[3] += 1
                truth_thick = truth >= 30
                pred_thick = estimate >= 30
                if pred_thick and truth_thick:
                    confusion[name][0] += 1
                elif pred_thick:
                    confusion[name][1] += 1
                elif truth_thick:
                    confusion[name][2] += 1

    if not scored_pairs or len(date_set) < 12:
        raise ValueError(f"insufficient outer support: pairs={len(scored_pairs)} dates={len(date_set)}")
    if sum(stratum_pixel_records.values()) != pixel_records:
        raise AssertionError("CPP-reference stratum support does not sum to the scored station-pixel records")
    dates = sorted(date_set)
    cells = [(site, lead) for site in SITES for lead in LEADS]
    methods = [*prediction, "persistence"]

    def score(name: str, draw: list[str], bucket: str = "all") -> tuple[float, float, float, int]:
        cell_rmse: list[float] = []
        cell_mae: list[float] = []
        cell_bias: list[float] = []
        total_n = 0
        for site, lead in cells:
            sq = ab = signed = n = 0.0
            for d in draw:
                v = stats.get((name, site, lead, d, bucket))
                if v:
                    sq += v[0]; ab += v[1]; signed += v[2]; n += v[3]
            if n:
                cell_rmse.append(math.sqrt(sq / n))
                cell_mae.append(ab / n)
                cell_bias.append(signed / n)
                total_n += int(n)
        if not cell_rmse:
            return float("nan"), float("nan"), float("nan"), 0
        return float(np.mean(cell_rmse)), float(np.mean(cell_mae)), float(np.mean(cell_bias)), total_n

    point = {m: score(m, dates) for m in methods}
    strata = ("near_zero_lt1", "thin_1_10", "moderate_10_30", "thick_ge30")
    thick = {m: score(m, dates, "thick_ge30") for m in methods}
    by_site = {m: {} for m in methods}
    # Recalculate each site-specific RMSE over 16 leads; do not pool site pixel counts.
    for m in methods:
        for site in SITES:
            vals = []
            for lead in LEADS:
                sq = n = 0.0
                for d in dates:
                    v = stats.get((m, site, lead, d, "all"))
                    if v:
                        sq += v[0]; n += v[3]
                if n:
                    vals.append(math.sqrt(sq / n))
            by_site[m][site] = float(np.mean(vals)) if vals else float("nan")
    by_lead = {m: {} for m in methods}
    for m in methods:
        for lead in LEADS:
            vals = []
            for site in SITES:
                sq = n = 0.0
                for d in dates:
                    v = stats.get((m, site, lead, d, "all"))
                    if v:
                        sq += v[0]; n += v[3]
                if n:
                    vals.append(math.sqrt(sq / n))
            by_lead[m][str(lead)] = float(np.mean(vals)) if vals else float("nan")

    seed_pairs = {
        seed: (f"oof_train_source_seed{seed}", f"fullr_control_seed{seed}")
        for seed in SEEDS
    }
    seed_deltas = {str(seed): point[oof][0] - point[full][0]
                   for seed, (oof, full) in seed_pairs.items()}
    mean_oof = float(np.mean([point[oof][0] for oof, _ in seed_pairs.values()]))
    mean_full = float(np.mean([point[full][0] for _, full in seed_pairs.values()]))
    mean_oof_by_site = {
        site: float(np.mean([by_site[oof][site] for oof, _ in seed_pairs.values()]))
        for site in SITES
    }
    mean_full_by_site = {
        site: float(np.mean([by_site[full][site] for _, full in seed_pairs.values()]))
        for site in SITES
    }
    mean_persistence = point["persistence"][0]
    seed_thick = {str(seed): thick[oof][0] - thick[full][0]
                  for seed, (oof, full) in seed_pairs.items()}
    mean_thick_oof = float(np.mean([thick[oof][0] for oof, _ in seed_pairs.values()]))
    mean_thick_full = float(np.mean([thick[full][0] for _, full in seed_pairs.values()]))
    mean_thick_persistence = thick["persistence"][0]
    mean_thick_oof_bias = float(np.mean([thick[oof][2] for oof, _ in seed_pairs.values()]))
    mean_thick_full_bias = float(np.mean([thick[full][2] for _, full in seed_pairs.values()]))
    mean_thick_persistence_bias = thick["persistence"][2]

    rng = np.random.default_rng(args.seed)
    boot_delta = np.empty(args.bootstrap, dtype=np.float64)
    boot_persistence = np.empty(args.bootstrap, dtype=np.float64)
    boot_delta_by_seed = {str(seed): np.empty(args.bootstrap, dtype=np.float64) for seed in SEEDS}
    for b in range(args.bootstrap):
        draw = rng.choice(dates, size=len(dates), replace=True).tolist()
        deltas = {}
        for seed, (oof, full) in seed_pairs.items():
            deltas[str(seed)] = score(oof, draw)[0] - score(full, draw)[0]
            boot_delta_by_seed[str(seed)][b] = deltas[str(seed)]
        boot_delta[b] = np.mean(list(deltas.values()))
        boot_persistence[b] = np.mean([
            score(oof, draw)[0] - score("persistence", draw)[0]
            for oof, _ in seed_pairs.values()
        ])

    detection = {}
    for name, (tp, fp, fn) in confusion.items():
        detection[name] = {
            "threshold_cot_ge30": True, "tp": tp, "fp": fp, "fn": fn,
            "pod": tp / (tp + fn) if tp + fn else None,
            "far": fp / (tp + fp) if tp + fp else None,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else None,
        }

    result = {
        "state": "COMPLETE_FP64_OOF_SOURCE_TRAIN_DATE_HOLDOUT_SCORE",
        "scope": "Hubei three station-center pixels; outer holdout from frozen original train dates only; CPP retrieval reference, not independent cloud truth",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "official_monthly_validation_payloads_read": 0,
        "station_center_pixels_row_col": SITES,
        "init_keys_common": len(common_init), "init_lead_pairs_scored": len(scored_pairs),
        "station_pixel_records": pixel_records, "target_dates_bjt": dates,
        "cpp_reference_stratum_station_pixel_records": stratum_pixel_records,
        "cpp_reference_stratum_target_dates": {name: len(ds) for name, ds in stratum_date_support.items()},
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
        "physical_cot_metrics_equal_station_equal_lead": {
            m: {"rmse": point[m][0], "mae": point[m][1], "bias": point[m][2], "records_across_cells": point[m][3]}
            for m in methods
        },
        "seed_pair_oof_minus_fullr_rmse": seed_deltas,
        "seed_pair_oof_minus_fullr_date_block_95pct_ci": {
            seed: np.quantile(values, [0.025, 0.975]).tolist()
            for seed, values in boot_delta_by_seed.items()
        },
        "seed_mean_rmse": {"oof_train_source": mean_oof, "fullr_control": mean_full, "persistence": mean_persistence},
        "seed_mean_oof_minus_fullr_rmse": mean_oof - mean_full,
        "seed_mean_oof_minus_fullr_date_block_95pct_ci": np.quantile(boot_delta, [0.025, 0.975]).tolist(),
        "seed_mean_oof_minus_persistence_rmse": mean_oof - mean_persistence,
        "seed_mean_oof_minus_persistence_date_block_95pct_ci": np.quantile(boot_persistence, [0.025, 0.975]).tolist(),
        "per_site_equal_lead_rmse": by_site,
        "seed_mean_per_site_equal_lead_rmse": {
            "oof_train_source": mean_oof_by_site,
            "fullr_control": mean_full_by_site,
        },
        "per_lead_equal_station_rmse": by_lead,
        "truth_cpp_strata_equal_station_equal_lead_rmse": {
            s: {m: score(m, dates, s)[0] for m in methods} for s in strata
        },
        "thick_cpp_ge30_rmse": {
            "seed_mean_oof_train_source": mean_thick_oof,
            "seed_mean_fullr_control": mean_thick_full,
            "persistence": mean_thick_persistence,
            "seed_pair_oof_minus_fullr": seed_thick,
            "seed_mean_oof_minus_fullr": mean_thick_oof - mean_thick_full,
            "seed_mean_bias": {
                "oof_train_source": mean_thick_oof_bias,
                "fullr_control": mean_thick_full_bias,
                "persistence": mean_thick_persistence_bias,
            },
        },
        "thick_detection_ge30": detection,
        "out_of_range_station_pixel_predictions_lt0_or_gt100": out_of_range,
        "paired_gate": {
            "seed_mean_point_improves": mean_oof < mean_full,
            "date_block_ci_entirely_below_zero": bool(np.quantile(boot_delta, 0.975) < 0),
            "at_least_two_of_three_seeds_improve": sum(v < 0 for v in seed_deltas.values()) >= 2,
            "no_site_regresses_more_than_5pct": all(mean_oof_by_site[s] <= 1.05 * mean_full_by_site[s] for s in SITES),
            "thick_rmse_no_worse_than_control_or_persistence": mean_thick_oof <= min(mean_thick_full, mean_thick_persistence),
        },
        "stratum_note": "CPP thickness bins are reference-product strata; no claim of independent COT truth.",
    }
    result["paired_gate"]["passes_all_predeclared_gates"] = all(result["paired_gate"].values())
    if not all(math.isfinite(float(v)) for v in boot_delta) or not all(math.isfinite(float(v)) for v in boot_persistence):
        raise ValueError("non-finite bootstrap result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.output.with_name("seed_pair_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "oof_rmse", "fullr_rmse", "delta_oof_minus_fullr", "oof_thick_rmse", "fullr_thick_rmse", "delta_thick"])
        for seed, (oof, full) in seed_pairs.items():
            writer.writerow([seed, point[oof][0], point[full][0], seed_deltas[str(seed)], thick[oof][0], thick[full][0], seed_thick[str(seed)]])
    print(json.dumps({k: result[k] for k in (
        "state", "init_keys_common", "init_lead_pairs_scored", "station_pixel_records",
        "target_dates_bjt", "seed_mean_rmse", "seed_mean_oof_minus_fullr_rmse",
        "seed_mean_oof_minus_fullr_date_block_95pct_ci", "thick_cpp_ge30_rmse", "paired_gate"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
