#!/usr/bin/env python3
"""Independent, validation-only recalculation for the frozen Hubei FA/FAC runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hubei_forecast_metrics import forecast_metrics, metrics_by_lead

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)
ARMS = ("FA", "FAC")
CLEAR_SKY_THRESHOLD_WM2 = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_day(value: Any) -> str:
    text = str(value)
    digits = "".join(character for character in text[:10] if character.isdigit())
    if len(digits) < 8:
        raise ValueError(f"invalid validation target timestamp: {text!r}")
    return digits[:8]


def _validation_dates(split_path: Path) -> set[str]:
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"date_BJT", "split"}.issubset(reader.fieldnames):
            raise ValueError("split must contain date_BJT and split columns")
        dates = {
            str(row["date_BJT"]).strip().replace("-", "")
            for row in reader
            if str(row["split"]).strip().lower() == "val"
        }
    if not dates or any(len(day) != 8 or not day.isdigit() for day in dates):
        raise ValueError("frozen split has no valid validation dates")
    return dates


def recalculate(validation_root: Path, split_path: Path) -> dict[str, Any]:
    if not validation_root.is_dir():
        raise FileNotFoundError(validation_root)
    if any("test" in item.name.lower() or "buffer" in item.name.lower()
           for item in validation_root.rglob("*")):
        raise ValueError("validation input root contains a test/buffer-named path")
    allowed_dates = _validation_dates(split_path)
    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    rows_total = 0

    for seed in SEEDS:
        run = validation_root / f"forecast_FA_FAC_seed{seed}_normreload_v2"
        if not run.is_dir():
            raise FileNotFoundError(run)
        for station in STATIONS:
            source_hashes[f"seed{seed}/{station}"] = {}
            folder = run / station
            for arm in ARMS:
                bundle = folder / f"{arm}_validation_predictions_with_keys.npz"
                metadata_path = folder / f"{arm}_validation.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                bundle_sha = sha256_file(bundle)
                metadata_sha = sha256_file(metadata_path)
                if bundle_sha != metadata.get("prediction_bundle_sha256"):
                    raise ValueError(f"prediction bundle SHA mismatch: {bundle}")
                if (metadata.get("input_mode") != "forecast"
                        or int(metadata.get("test_payloads_read", -1)) != 0
                        or metadata.get("station") != station
                        or int(metadata.get("seed", -1)) != seed
                        or metadata.get("arm") != arm):
                    raise ValueError(f"validation provenance check failed: {metadata_path}")

                with np.load(bundle, allow_pickle=False) as data:
                    required = {
                        "truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2",
                        "init_time_bjt", "target_time_bjt", "lead_minutes",
                    }
                    missing = required.difference(data.files)
                    if missing:
                        raise ValueError(f"{bundle} missing arrays: {sorted(missing)}")
                    truth = np.asarray(data["truth_ghi_wm2"], dtype=np.float64)
                    prediction = np.asarray(data["pred_ghi_wm2"], dtype=np.float64)
                    clear_sky = np.asarray(data["clear_sky_ghi_wm2"], dtype=np.float64)
                    leads = np.asarray(data["lead_minutes"])
                    init = np.asarray(data["init_time_bjt"]).astype(str)
                    target = np.asarray(data["target_time_bjt"]).astype(str)
                n = len(truth)
                if n == 0 or any(array.ndim != 1 or len(array) != n
                                 for array in (prediction, clear_sky, leads, init, target)):
                    raise ValueError(f"empty or inconsistent validation bundle: {bundle}")
                target_days = {_target_day(value) for value in target}
                if not target_days.issubset(allowed_dates):
                    raise ValueError(f"non-validation target date in {bundle}")
                if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not np.isfinite(clear_sky).all():
                    raise ValueError(f"non-finite data in {bundle}")
                if (truth < 0).any() or (clear_sky <= 0).any():
                    raise ValueError(f"invalid GHI/daylight support in {bundle}")

                keys = [(str(i), str(t), int(l)) for i, t, l in zip(init, target, leads)]
                if len(keys) != len(set(keys)):
                    raise ValueError(f"duplicate validation key in {bundle}")
                metrics = forecast_metrics(
                    truth, prediction, clear_sky_wm2=clear_sky,
                    clear_sky_threshold_wm2=CLEAR_SKY_THRESHOLD_WM2,
                )
                by_lead = metrics_by_lead(
                    truth, prediction, leads, clear_sky_wm2=clear_sky,
                    clear_sky_threshold_wm2=CLEAR_SKY_THRESHOLD_WM2,
                )
                records[(seed, station, arm)] = {
                    "keys": keys, "truth": truth, "clear_sky": clear_sky,
                    "prediction": prediction, "metrics": metrics,
                }
                source_hashes[f"seed{seed}/{station}"][arm] = {
                    "bundle_sha256": bundle_sha,
                    "metadata_sha256": metadata_sha,
                    "n": n,
                    "dates": sorted(target_days),
                    "metrics": metrics,
                    "per_lead": {str(lead): value for lead, value in by_lead.items()},
                }
                rows_total += n

    for station in STATIONS:
        reference = records[(SEEDS[0], station, ARMS[0])]
        for seed in SEEDS:
            for arm in ARMS:
                current = records[(seed, station, arm)]
                for field in ("keys",):
                    if current[field] != reference[field]:
                        raise ValueError(f"paired validation keys differ: seed={seed}, {station}, {arm}")
                for field in ("truth", "clear_sky"):
                    if not np.array_equal(current[field], reference[field]):
                        raise ValueError(f"paired validation {field} differs: seed={seed}, {station}, {arm}")

    per_seed = {
        seed: {
            arm: float(np.mean([records[(seed, station, arm)]["metrics"]["rmse_wm2"]
                                for station in STATIONS], dtype=np.float64))
            for arm in ARMS
        }
        for seed in SEEDS
    }
    primary = {
        arm: float(np.mean([per_seed[seed][arm] for seed in SEEDS], dtype=np.float64))
        for arm in ARMS
    }
    key_sets = [set(records[(SEEDS[0], station, ARMS[0])]["keys"]) for station in STATIONS]
    common_keys = sorted(set.intersection(*key_sets))
    common_support: dict[str, Any] = {
        "n_common_keys": len(common_keys),
        "target_dates_bjt": sorted({_target_day(key[1]) for key in common_keys}),
    }
    if common_keys:
        common_per_seed: dict[int, dict[str, float]] = {}
        common_per_group: dict[str, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            common_per_seed[seed] = {}
            for arm in ARMS:
                station_scores = []
                for station in STATIONS:
                    rec = records[(seed, station, arm)]
                    index_by_key = {key: index for index, key in enumerate(rec["keys"])}
                    indices = np.asarray([index_by_key[key] for key in common_keys], dtype=np.int64)
                    metrics = forecast_metrics(
                        rec["truth"][indices], rec["prediction"][indices],
                        clear_sky_wm2=rec["clear_sky"][indices],
                        clear_sky_threshold_wm2=CLEAR_SKY_THRESHOLD_WM2,
                    )
                    station_scores.append(metrics["rmse_wm2"])
                    common_per_group.setdefault(station, {})[f"seed{seed}_{arm}"] = metrics
                common_per_seed[seed][arm] = float(np.mean(station_scores, dtype=np.float64))
        common_support["per_seed_station_equal_rmse_wm2"] = common_per_seed
        common_support["primary_rmse_wm2"] = {
            arm: float(np.mean([common_per_seed[seed][arm] for seed in SEEDS], dtype=np.float64))
            for arm in ARMS
        }
        common_support["FAC_minus_FA_rmse_wm2"] = (
            common_support["primary_rmse_wm2"]["FAC"] - common_support["primary_rmse_wm2"]["FA"]
        )
        common_support["per_station_seed_metrics"] = common_per_group
    else:
        common_support["status"] = "no_three_station_common_keys"
    return {
        "state": "HUBEI_FORECAST_VALIDATION_INDEPENDENT_RECALC_COMPLETE_TEST_CLOSED",
        "scope": "frozen validation only; not independent test or confirmatory paper evidence",
        "validation_root_name": validation_root.name,
        "split_sha256": sha256_file(split_path),
        "validation_dates_bjt": sorted(allowed_dates),
        "stations": list(STATIONS), "seeds": list(SEEDS), "arms": list(ARMS),
        "rows_total_across_bundles": rows_total,
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "clear_sky_sensitivity_threshold_wm2": CLEAR_SKY_THRESHOLD_WM2,
        "primary_rmse_wm2": primary,
        "FAC_minus_FA_rmse_wm2": primary["FAC"] - primary["FA"],
        "per_seed_station_equal_rmse_wm2": per_seed,
        "per_station_seed_mean_rmse_wm2": {
            station: {
                arm: float(np.mean([records[(seed, station, arm)]["metrics"]["rmse_wm2"]
                                    for seed in SEEDS], dtype=np.float64))
                for arm in ARMS
            }
            for station in STATIONS
        },
        "three_station_common_support_sensitivity": common_support,
        "source_bundles": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True,
                        help="Directory containing the three seed*_normreload_v2 validation runs")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = recalculate(args.validation_root, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(args.output.name + ".tmp")
    temp_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp_output, args.output)
    print(json.dumps({key: report[key] for key in (
        "state", "primary_rmse_wm2", "FAC_minus_FA_rmse_wm2",
        "rows_total_across_bundles", "test_payloads_read", "buffer_payloads_read",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
