#!/usr/bin/env python3
"""Freeze illustrative Hubei validation dates without inspecting model errors.

The selected "change" example is explicitly an observed-GHI ramp case, not a
cloud-weather label. Selection reads only the validation labels/masks. It never
loads A/AC/AP predictions, CPP/COT values, test rows, or buffer rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter_DJF"
    if month in (3, 4, 5):
        return "spring_MAM"
    if month in (6, 7, 8):
        return "summer_JJA"
    return "autumn_SON"


def row_valid(row: dict[str, str]) -> bool:
    return (
        row.get("split") == "val"
        and row.get("valid_daylight_clear_sky_label") == "1"
        and row.get("mask") == "1"
        and row.get("ghi15_raw", "") != ""
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    labels = args.labels.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_day: dict[tuple[str, str], dict[datetime, float]] = defaultdict(dict)
    with labels.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "val":
                continue
            stamp = datetime.strptime(row["target_time_BJT"], "%Y-%m-%d %H:%M:%S")
            if row_valid(row):
                rows_by_day[(row["station"], stamp.date().isoformat())][stamp] = float(row["ghi15_raw"])

    by_season: dict[tuple[str, str], list[tuple[str, dict[datetime, float]]]] = defaultdict(list)
    for (station, day), values in rows_by_day.items():
        d = datetime.strptime(day, "%Y-%m-%d").date()
        by_season[(station, season(d.month))].append((day, values))

    selections: list[dict[str, str | int | float]] = []
    for station in ("Niushou", "Wangzhai", "Mizhuang"):
        for seas in ("winter_DJF", "spring_MAM", "summer_JJA", "autumn_SON"):
            candidates = sorted(by_season.get((station, seas), []), key=lambda x: x[0])
            # A ramp is the largest range over any 5 consecutive valid 15-min
            # daylight targets within a day. This uses observations only.
            scored: list[tuple[float, str, int]] = []
            for day, values in candidates:
                score = float("-inf")
                stamps = sorted(values)
                for i in range(len(stamps) - 4):
                    window = stamps[i : i + 5]
                    if all(window[j] - window[j - 1] == timedelta(minutes=15) for j in range(1, 5)):
                        vals = [values[t] for t in window]
                        score = max(score, max(vals) - min(vals))
                if score != float("-inf"):
                    scored.append((score, day, len(values)))

            if scored:
                score, event_day, event_n = sorted(scored, key=lambda x: (-x[0], x[1]))[0]
                selections.append({
                    "station": station,
                    "season": seas,
                    "case_type": "largest_observed_GHI_60min_range",
                    "date_BJT": event_day,
                    "valid_daylight_targets": event_n,
                    "observed_GHI_range_wm2": round(score, 6),
                    "selection_basis": "truth_and_mask_only_no_model_prediction_or_error",
                    "weather_class": "unverified_no_independent_CPP_CLP_label",
                })
                remaining = [x for x in candidates if x[0] != event_day]
            else:
                remaining = candidates
                selections.append({
                    "station": station,
                    "season": seas,
                    "case_type": "largest_observed_GHI_60min_range",
                    "date_BJT": "",
                    "valid_daylight_targets": 0,
                    "observed_GHI_range_wm2": "",
                    "selection_basis": "not_selected_insufficient_five_consecutive_valid_15min_targets",
                    "weather_class": "unverified_no_independent_CPP_CLP_label",
                })

            if remaining:
                date_hashes = [
                    (hashlib.sha256(f"hubei-val-case-v1|{station}|{seas}|{day}".encode()).hexdigest(), day, values)
                    for day, values in remaining
                ]
                digest, fixed_day, fixed_values = min(date_hashes, key=lambda x: x[0])
                selections.append({
                    "station": station,
                    "season": seas,
                    "case_type": "fixed_hash_date",
                    "date_BJT": fixed_day,
                    "valid_daylight_targets": len(fixed_values),
                    "observed_GHI_range_wm2": "",
                    "selection_basis": "minimum_SHA256_of_station_season_date_among_remaining_valid_validation_dates",
                    "selection_hash": digest,
                    "weather_class": "unverified_no_independent_CPP_CLP_label",
                })
            else:
                selections.append({
                    "station": station,
                    "season": seas,
                    "case_type": "fixed_hash_date",
                    "date_BJT": "",
                    "valid_daylight_targets": 0,
                    "observed_GHI_range_wm2": "",
                    "selection_basis": "no_remaining_validation_date_after_change_case_or_no_support",
                    "selection_hash": "",
                    "weather_class": "unverified_no_independent_CPP_CLP_label",
                })

    manifest_path = out_dir / "validation_case_manifest.csv"
    fields = [
        "station", "season", "case_type", "date_BJT", "valid_daylight_targets",
        "observed_GHI_range_wm2", "selection_basis", "selection_hash", "weather_class",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selections)

    audit = {
        "state": "FROZEN_VALIDATION_CASE_SELECTION_NO_MODEL_ERRORS_READ",
        "input_labels": str(labels),
        "input_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "selection_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "split_filter": "val_only",
        "eligible_label_filter": "valid_daylight_clear_sky_label==1 and mask==1",
        "change_case": "largest observed GHI range over five consecutive 15-minute targets (60 minutes); illustrative ramp, not weather class",
        "fixed_case": "min SHA256(hubei-val-case-v1|station|season|date) among remaining supported validation dates",
        "model_predictions_or_errors_read": False,
        "CPP_or_COT_read_for_selection": False,
        "test_or_buffer_rows_read": False,
        "cloud_weather_labels": "unavailable; no independent CPP/CLP class assigned",
        "selection_count": sum(bool(x.get("date_BJT")) for x in selections),
        "records": selections,
    }
    audit_path = out_dir / "validation_case_selection_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "state": audit["state"],
        "manifest": str(manifest_path),
        "manifest_sha256": audit["output_manifest_sha256"],
        "audit": str(audit_path),
        "selected_dates": audit["selection_count"],
        "test_or_buffer_rows_read": False,
        "model_predictions_or_errors_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
