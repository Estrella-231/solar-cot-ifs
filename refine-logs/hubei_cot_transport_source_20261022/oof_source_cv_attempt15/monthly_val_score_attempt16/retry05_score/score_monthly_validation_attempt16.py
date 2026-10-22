#!/usr/bin/env python3
"""FP64 one-time score of frozen attempt15 checkpoints on monthly validation."""
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
SEEDS = (42, 43, 44)
ARMS = ("fullr_control", "oof_train_source")
LOG_DENOM = math.log(101.0)
STRATA = ("near_zero_lt1", "thin_1_10", "moderate_10_30", "thick_ge30")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y%m%d%H%M")


def load_prediction(path: Path, expected_name: str,
                    checkpoint_root: Path) -> tuple[dict[str, np.ndarray], dict]:
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("state") != "COMPLETE_FROZEN_MONTHLY_VALIDATION_INFERENCE":
        raise ValueError(f"inference receipt is incomplete: {receipt_path}")
    if receipt.get("model_name") != expected_name:
        raise ValueError(f"model name mismatch in {receipt_path}")
    if receipt.get("prediction_sha256") != sha256(path):
        raise ValueError(f"prediction hash mismatch: {path}")
    if (receipt.get("test_payloads_read") != 0 or receipt.get("buffer_payloads_read") != 0
            or receipt.get("optimizer_updates") != 0
            or receipt.get("future_cpp_values_or_masks_loaded_for_inference") is not False):
        raise ValueError(f"inference receipt violates frozen causal protocol: {receipt_path}")
    if receipt.get("official_monthly_validation_cache_used_for_inference") is not True:
        raise ValueError(f"inference did not use the declared monthly validation cache: {receipt_path}")

    checkpoint = Path(receipt["checkpoint"])
    if not checkpoint.is_absolute() or checkpoint.parent.parent != checkpoint_root.resolve():
        raise ValueError(f"checkpoint path is outside the declared attempt15 root: {checkpoint}")
    run_name = checkpoint.parent.name
    complete_path = checkpoint.parent / "training_complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    seed = int(expected_name.rsplit("seed", 1)[1])
    if (complete.get("state") != "COMPLETE_TRAINED_HUBEI_INTERNAL_TRAIN_DATE_HOLDOUT_COT_DIAGNOSTIC"
            or complete.get("seed") != seed
            or complete.get("global_step") != 1350
            or complete.get("global_batch") != 32
            or complete.get("world_size") != 1
            or complete.get("train_config", {}).get("width") != 96
            or complete.get("official_monthly_validation_payloads_read", 0) != 0
            or "official monthly validation, test, and buffer not read" not in complete.get("split", "")
            or complete.get("test_payloads_read") != 0
            or complete.get("buffer_payloads_read") != 0):
        raise ValueError(f"training receipt is outside the attempt15 frozen contract: {complete_path}")
    if run_name != expected_name or complete.get("best_checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"checkpoint provenance/hash mismatch: {checkpoint}")
    if receipt.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"inference checkpoint hash mismatch: {checkpoint}")

    archive = np.load(path, allow_pickle=False)
    times = [str(x) for x in archive["init_time_bjt"].tolist()]
    pred = np.asarray(archive["pred_log_scaled"], dtype=np.float64)
    if pred.shape != (len(times), 16, 1, 16, 16) or len(times) != len(set(times)):
        raise ValueError(f"bad prediction schema or duplicate keys in {path}: {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"non-finite prediction in {path}")
    return {key: pred[i] for i, key in enumerate(times)}, receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--prediction-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261022)
    args = ap.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("at least 1000 date-block bootstrap draws are required")

    cache = args.cache.resolve()
    report_path = cache / "cache_report.json"
    audit_path = cache.parent / "cache_audit" / "cache_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_HUBEI_STATION_PATCH_COT_GEOMETRY_CACHE_TRAINVAL_ONLY":
        raise ValueError("unexpected cache state")
    if audit.get("state") != "PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED":
        raise ValueError("monthly cache audit did not pass")
    for record in (report, audit):
        if record.get("test_payloads_read") != 0 or record.get("buffer_payloads_read") != 0:
            raise ValueError("test/buffer payload gate is not closed")
    if report.get("split", {}).get("test_payloads_read") != 0 or report.get("split", {}).get("buffer_payloads_read") != 0:
        raise ValueError("nested test/buffer payload gate is not closed")
    if report.get("split", {}).get("val") != 1855:
        raise ValueError(f"unexpected monthly validation frame count: {report.get('split')}")

    bank_rows = read_csv(cache / "bank_rows.csv")
    bank_index: dict[str, int] = {}
    split_by_time: dict[str, str] = {}
    for row in bank_rows:
        key = row["time_bjt"]
        split = row["split"].lower()
        if split not in {"train", "val"}:
            raise ValueError(f"unexpected split in train/val-only cache: {split}")
        if key in bank_index:
            raise ValueError(f"duplicate bank time: {key}")
        bank_index[key] = int(row["index"])
        split_by_time[key] = split

    val_pairs = [r for r in read_csv(cache / "forecast_pairs.csv") if r["split"].lower() == "val"]
    if len(val_pairs) != 4116:
        raise ValueError(f"unexpected monthly validation pair count: {len(val_pairs)}")
    expected_init = set()
    seen_pairs = set()
    for row in val_pairs:
        init_s, target_s = row["init_time_bjt"], row["target_time_bjt"]
        lead = int(row["lead_minutes"])
        if lead not in LEADS or parse_time(target_s) - parse_time(init_s) != dt.timedelta(minutes=lead):
            raise ValueError(f"invalid validation pair key: {row}")
        key = (init_s, target_s, lead)
        if key in seen_pairs:
            raise ValueError(f"duplicate monthly-validation init/target/lead key: {key}")
        seen_pairs.add(key)
        if split_by_time.get(init_s) != "val" or split_by_time.get(target_s) != "val":
            raise ValueError(f"cross-split monthly validation pair: {row}")
        if bank_index.get(init_s) != int(row["init_index"]) or bank_index.get(target_s) != int(row["target_index"]):
            raise ValueError(f"bank index mismatch: {row}")
        expected_init.add(init_s)
    if len(expected_init) != 786:
        raise ValueError(f"unexpected validation init-window count: {len(expected_init)}")

    prediction: dict[str, dict[str, np.ndarray]] = {}
    prediction_receipts = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_seed{seed}"
            path = args.prediction_root / f"{name}.npz"
            values, receipt = load_prediction(path, name, args.training_root)
            if set(values) != expected_init:
                raise ValueError(f"prediction key set differs from monthly val init set in {name}")
            prediction[name] = values
            prediction_receipts[name] = receipt
    common_init = set.intersection(*(set(v) for v in prediction.values()))
    if common_init != expected_init:
        raise ValueError("six frozen models do not share the complete validation init-key set")

    cache_arrays = {
        name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("history_cot_log", "history_valid", "cpp_cot", "cpp_valid")
    }
    # (method, site, lead, target_date, stratum) -> [sum squared, abs, signed, count]
    stats: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(lambda: [0., 0., 0., 0.])
    target_dates: set[str] = set()
    scored_keys: set[tuple[str, str, int]] = set()
    stratum_dates = {name: set() for name in STRATA}
    stratum_records = {name: 0 for name in STRATA}
    confusion = {name: [0, 0, 0] for name in [*prediction, "persistence"]}
    by_time = bank_index

    for row in val_pairs:
        init_s, target_s = row["init_time_bjt"], row["target_time_bjt"]
        lead = int(row["lead_minutes"])
        init_i, target_i = by_time[init_s], by_time[target_s]
        li = lead // 15 - 1
        date = parse_time(target_s).strftime("%Y%m%d")
        history_log = np.asarray(cache_arrays["history_cot_log"][init_i, 0], dtype=np.float64)
        history_valid = np.asarray(cache_arrays["history_valid"][init_i, 0], dtype=bool)
        persistence = np.expm1(history_log * LOG_DENOM)
        target_map = np.asarray(cache_arrays["cpp_cot"][target_i, 0], dtype=np.float64)
        target_valid = np.asarray(cache_arrays["cpp_valid"][target_i, 0], dtype=bool)
        actual_mask_count = int(target_valid.sum())
        if actual_mask_count != int(row["target_valid_pixels"]):
            raise ValueError(f"target mask count mismatch at {target_s}")

        for site, (rr, cc) in SITES.items():
            if not (target_valid[rr, cc] and history_valid[rr, cc]):
                continue
            target_dates.add(date)
            truth = float(target_map[rr, cc])
            if not math.isfinite(truth) or truth < 0 or truth > 100:
                raise ValueError(f"invalid CPP retrieval reference at {target_s}/{site}: {truth}")
            scored_keys.add((init_s, target_s, lead))
            stratum = "near_zero_lt1" if truth < 1 else "thin_1_10" if truth < 10 else "moderate_10_30" if truth < 30 else "thick_ge30"
            stratum_records[stratum] += 1
            stratum_dates[stratum].add(date)

            estimates: dict[str, float] = {}
            for name, maps in prediction.items():
                estimate_map = np.expm1(maps[init_s][li, 0] * LOG_DENOM)
                estimate = float(estimate_map[rr, cc])
                if not math.isfinite(estimate):
                    raise ValueError(f"non-finite prediction: {name} {init_s} +{lead}")
                estimates[name] = estimate
            estimates["persistence"] = float(persistence[rr, cc])
            for name, estimate in estimates.items():
                error = estimate - truth
                for bucket in ("all", stratum):
                    acc = stats[(name, site, lead, date, bucket)]
                    acc[0] += error * error
                    acc[1] += abs(error)
                    acc[2] += error
                    acc[3] += 1
                pred_event, truth_event = estimate >= 30, truth >= 30
                if pred_event and truth_event:
                    confusion[name][0] += 1
                elif pred_event:
                    confusion[name][1] += 1
                elif truth_event:
                    confusion[name][2] += 1

    dates = sorted(target_dates)
    if len(dates) < 12 or len(scored_keys) < 1000:
        raise ValueError(f"insufficient common validation support: dates={len(dates)} pairs={len(scored_keys)}")
    if sum(stratum_records.values()) != sum(v[3] for k, v in stats.items() if k[-1] == "all") // len([*prediction, "persistence"]):
        raise AssertionError("stratum and full support accounting mismatch")

    cells = [(site, lead) for site in SITES for lead in LEADS]

    def score(method: str, draw_dates: list[str], bucket: str = "all",
              site_filter: str | None = None, lead_filter: int | None = None) -> tuple[float, int]:
        values = []
        count = 0
        active_cells = [(s, l) for s, l in cells
                        if (site_filter is None or s == site_filter)
                        and (lead_filter is None or l == lead_filter)]
        for site, lead in active_cells:
            sq = n = 0.0
            for date in draw_dates:
                cell = stats.get((method, site, lead, date, bucket))
                if cell:
                    sq += cell[0]
                    n += cell[3]
            if n:
                values.append(math.sqrt(sq / n))
                count += int(n)
        return (float(np.mean(values)) if values else float("nan"), count)

    def mae_bias(method: str, draw_dates: list[str], bucket: str = "all") -> tuple[float, float]:
        mae_cells = []
        bias_cells = []
        for site, lead in cells:
            ab = signed = n = 0.0
            for date in draw_dates:
                cell = stats.get((method, site, lead, date, bucket))
                if cell:
                    ab += cell[1]
                    signed += cell[2]
                    n += cell[3]
            if n:
                mae_cells.append(ab / n)
                bias_cells.append(signed / n)
        return (float(np.mean(mae_cells)) if mae_cells else float("nan"),
                float(np.mean(bias_cells)) if bias_cells else float("nan"))

    def arm_score(arm: str, draw_dates: list[str], bucket: str = "all",
                  site_filter: str | None = None,
                  lead_filter: int | None = None) -> float:
        per_seed = [score(f"{arm}_seed{seed}", draw_dates, bucket, site_filter, lead_filter)[0]
                    for seed in SEEDS]
        return float(np.mean(per_seed))

    methods = [*prediction, "persistence"]
    point = {name: score(name, dates)[0] for name in methods}
    arm_point = {arm: arm_score(arm, dates) for arm in ARMS}
    persistence_point = point["persistence"]
    per_site = {arm: {site: arm_score(arm, dates, site_filter=site) for site in SITES} for arm in ARMS}
    per_lead = {arm: {str(lead): arm_score(arm, dates, lead_filter=lead) for lead in LEADS} for arm in ARMS}
    per_stratum = {bucket: {name: score(name, dates, bucket)[0] for name in methods}
                   for bucket in STRATA}
    per_method_mae_bias = {name: {"mae": mae_bias(name, dates)[0], "bias": mae_bias(name, dates)[1]}
                           for name in methods}
    per_arm_mae_bias = {
        arm: {metric: float(np.mean([per_method_mae_bias[f"{arm}_seed{seed}"][metric] for seed in SEEDS]))
              for metric in ("mae", "bias")}
        for arm in ARMS
    }
    thick = per_stratum["thick_ge30"]
    seed_deltas = {
        str(seed): point[f"oof_train_source_seed{seed}"] - point[f"fullr_control_seed{seed}"]
        for seed in SEEDS
    }

    rng = np.random.default_rng(args.bootstrap_seed)
    boot_oof_full = np.empty(args.bootstrap, dtype=np.float64)
    boot_oof_persist = np.empty(args.bootstrap, dtype=np.float64)
    boot_seed_delta = {str(seed): np.empty(args.bootstrap, dtype=np.float64) for seed in SEEDS}
    for b in range(args.bootstrap):
        draw = rng.choice(dates, size=len(dates), replace=True).tolist()
        for seed in SEEDS:
            delta = score(f"oof_train_source_seed{seed}", draw)[0] - score(f"fullr_control_seed{seed}", draw)[0]
            boot_seed_delta[str(seed)][b] = delta
        boot_oof_full[b] = arm_score("oof_train_source", draw) - arm_score("fullr_control", draw)
        boot_oof_persist[b] = arm_score("oof_train_source", draw) - score("persistence", draw)[0]

    c95 = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    detection = {}
    for name, (tp, fp, fn) in confusion.items():
        detection[name] = {
            "reference_threshold_cot_ge30": True, "tp": tp, "fp": fp, "fn": fn,
            "pod": tp / (tp + fn) if tp + fn else None,
            "far": fp / (tp + fp) if tp + fp else None,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else None,
        }

    gates = {
        "oof_seed_mean_rmse_improves_vs_fullr": arm_point["oof_train_source"] < arm_point["fullr_control"],
        "oof_seed_mean_rmse_improves_vs_persistence": arm_point["oof_train_source"] < persistence_point,
        "date_block_95pct_ci_oof_minus_fullr_entirely_below_zero": bool(np.quantile(boot_oof_full, 0.975) < 0),
        "all_three_seed_rmse_directions_favor_oof": all(v < 0 for v in seed_deltas.values()),
        "no_station_regresses_more_than_5pct": all(per_site["oof_train_source"][site] <= 1.05 * per_site["fullr_control"][site] for site in SITES),
        "thick_reference_rmse_no_worse_than_fullr_or_persistence": arm_score("oof_train_source", dates, "thick_ge30") <= min(arm_score("fullr_control", dates, "thick_ge30"), score("persistence", dates, "thick_ge30")[0]),
    }
    gates["passes_all_predeclared_gates"] = all(gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "monthly_validation_score.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing score: {output}")
    result = {
        "state": "COMPLETE_FP64_FROZEN_ATTEMPT15_MONTHLY_VALIDATION_SCORE",
        "scope": "Hubei three station-center pixels on the frozen monthly validation split; retrospective validation diagnostic, not independent test",
        "reference_role": "CPP is a retrieval reference, not independent physical truth",
        "input_mode": "8-frame causal historical R-COT sequence plus deterministic geometry; future CPP values and masks are score-only",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "monthly_validation_cache_used_for_inference": True,
        "future_cpp_values_or_masks_loaded_for_inference": False,
        "cpp_reference_arrays_read_for_offline_scoring": ["cpp_cot.npy", "cpp_valid.npy"],
        "optimizer_updates": 0,
        "cache_report_sha256": sha256(report_path), "cache_audit_sha256": sha256(audit_path),
        "scorer_source_sha256": sha256(Path(__file__).resolve()),
        "inference_source_sha256": sha256(Path(__file__).with_name("infer_frozen_monthly_validation.py")),
        "prediction_receipts": {name: {"prediction_sha256": rec["prediction_sha256"],
                                          "checkpoint_sha256": rec["checkpoint_sha256"],
                                          "device": rec["device"],
                                          "inference_script_sha256": rec["inference_script_sha256"]}
                                for name, rec in prediction_receipts.items()},
        "station_center_pixels_row_col": SITES,
        "monthly_validation_init_windows": len(common_init),
        "init_lead_pairs_in_validation_cache": len(val_pairs),
        "init_lead_pairs_with_any_valid_station_pixel": len(scored_keys),
        "station_pixel_records": int(sum(v[3] for k, v in stats.items() if k[-1] == "all") // len(methods)),
        "target_dates_bjt": dates,
        "records_by_cpp_reference_stratum": stratum_records,
        "dates_by_cpp_reference_stratum": {k: sorted(v) for k, v in stratum_dates.items()},
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "physical_cot_rmse_equal_station_equal_lead": {
            **{name: point[name] for name in methods},
            "fullr_control_seed_mean": arm_point["fullr_control"],
            "oof_train_source_seed_mean": arm_point["oof_train_source"],
        },
        "physical_cot_mae_and_bias_equal_station_equal_lead": {
            **per_method_mae_bias,
            "fullr_control_seed_mean": per_arm_mae_bias["fullr_control"],
            "oof_train_source_seed_mean": per_arm_mae_bias["oof_train_source"],
        },
        "oof_minus_fullr_rmse_seed_pair": seed_deltas,
        "oof_minus_fullr_rmse_seed_mean": arm_point["oof_train_source"] - arm_point["fullr_control"],
        "oof_minus_fullr_rmse_seed_mean_date_block_95pct_ci": c95(boot_oof_full),
        "oof_minus_fullr_rmse_seed_pair_date_block_95pct_ci": {k: c95(v) for k, v in boot_seed_delta.items()},
        "oof_minus_persistence_rmse_seed_mean": arm_point["oof_train_source"] - persistence_point,
        "oof_minus_persistence_rmse_date_block_95pct_ci": c95(boot_oof_persist),
        "per_station_equal_lead_rmse": per_site,
        "per_lead_equal_station_rmse": per_lead,
        "cpp_reference_stratum_equal_station_equal_lead_rmse": per_stratum,
        "thick_reference_cot_ge30_detection": detection,
        "gate": gates,
        "interpretation_limit": "This single retrospective monthly-validation score can inform the next training hypothesis but is neither independent test nor evidence of GHI benefit. Do not tune or retrain attempt15 on this cohort.",
    }

    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "common_scored_keys.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["init_time_bjt", "target_time_bjt", "lead_minutes"])
        for init_s, target_s, lead in sorted(scored_keys):
            writer.writerow([init_s, target_s, lead])
    print(json.dumps({"state": result["state"], "windows": len(common_init),
                      "pairs": len(scored_keys), "records": result["station_pixel_records"],
                      "dates": dates,
                      "rmse": result["physical_cot_rmse_equal_station_equal_lead"],
                      "delta": result["oof_minus_fullr_rmse_seed_mean"],
                      "ci95": result["oof_minus_fullr_rmse_seed_mean_date_block_95pct_ci"],
                      "gate": gates}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
