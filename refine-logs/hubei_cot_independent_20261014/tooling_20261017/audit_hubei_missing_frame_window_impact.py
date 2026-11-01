#!/usr/bin/env python3
"""Metadata-only upper bound on same-day 24-frame windows restored by AGRI gaps.

This reads only the frozen split and frame/request manifests. It never opens
AGRI arrays, station labels, CPP/COT arrays, predictions, or test/buffer data.
The count deliberately does not apply the per-frame daytime mask, so it is an
upper bound, not the number of samples that would enter training.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAME_MANIFEST = ROOT / "frames_manifest_20260601_20260714.csv"
SPLIT_MANIFEST = ROOT / "frozen_split_20261015_v1" / "day_assignment.csv"
REQUEST_MANIFEST = ROOT / "AGRI_ARCHIVE_REQUEST_16_SLOTS_20261018.csv"
WINDOW_FRAMES = 24
CADENCE_MINUTES = 15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_bjt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def count_windows(keys: set[str], day: datetime) -> int:
    count = 0
    # A BJT day contains 96 quarter-hour slots. Non-published/missing slots
    # naturally break windows because their keys are absent from `keys`.
    for start_index in range(96 - WINDOW_FRAMES + 1):
        window = [
            (day + timedelta(minutes=CADENCE_MINUTES * (start_index + offset))).strftime("%Y%m%d%H%M")
            for offset in range(WINDOW_FRAMES)
        ]
        count += int(all(key in keys for key in window))
    return count


def main() -> None:
    frame_rows = read_csv(FRAME_MANIFEST)
    split_rows = read_csv(SPLIT_MANIFEST)
    request_rows = read_csv(REQUEST_MANIFEST)

    present: set[str] = set()
    for row in frame_rows:
        key = datetime.strptime(row["time_bjt"], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")
        if key in present:
            raise ValueError(f"Duplicate BJT frame key: {key}")
        present.add(key)

    requested: set[str] = set()
    for row in request_rows:
        utc = datetime.fromisoformat(row["utc_start"].replace("Z", "+00:00"))
        # Parse the explicit BJT field and validate it against UTC+8 without
        # depending on the machine's local timezone.
        expected_bjt = (utc.replace(tzinfo=None) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
        if row["bjt_valid_time"] != expected_bjt:
            raise ValueError(f"UTC/BJT mismatch for request slot {row['utc_start']}")
        key = parse_bjt(row["bjt_valid_time"]).strftime("%Y%m%d%H%M")
        if key in present or key in requested:
            raise ValueError(f"Requested slot already present or duplicated: {key}")
        requested.add(key)

    split_by_day: dict[str, str] = {}
    for row in split_rows:
        day = row["date_BJT"].replace("-", "")
        if day in split_by_day:
            raise ValueError(f"Duplicate split date: {day}")
        split_by_day[day] = row["split"].strip().lower()

    if len(request_rows) != 16 or len(requested) != 16:
        raise ValueError(f"Expected 16 unique request slots, got {len(requested)}")
    request_days = {key[:8] for key in requested}
    invalid_days = {
        day: split_by_day.get(day, "missing")
        for day in request_days
        if split_by_day.get(day) not in {"train", "val", "validation"}
    }
    if invalid_days:
        raise ValueError(f"Requested slots not confined to train/validation: {invalid_days}")

    result_by_split: dict[str, Counter[str]] = {"train": Counter(), "validation": Counter()}
    affected_days: list[dict[str, int | str]] = []
    for day_str, split in sorted(split_by_day.items()):
        if not ("20260601" <= day_str <= "20260714") or split not in {"train", "val", "validation"}:
            continue
        label = "validation" if split in {"val", "validation"} else "train"
        day = datetime.strptime(day_str, "%Y%m%d")
        before = count_windows(present, day)
        after = count_windows(present | requested, day)
        result_by_split[label]["current"] += before
        result_by_split[label]["after_restore_upper_bound"] += after
        result_by_split[label]["added_upper_bound"] += after - before
        if after > before:
            affected_days.append({
                "date_BJT": day.strftime("%Y-%m-%d"),
                "split": label,
                "current_windows_without_day_mask": before,
                "windows_if_all_requested_frames_restored_without_day_mask": after,
                "added_upper_bound": after - before,
            })

    report = {
        "status": "METADATA_ONLY_UPPER_BOUND",
        "inputs": {
            "frame_manifest": FRAME_MANIFEST.name,
            "split_manifest": SPLIT_MANIFEST.name,
            "request_manifest": REQUEST_MANIFEST.name,
        },
        "input_rows": {
            "accepted_frames": len(frame_rows),
            "requested_missing_slots": len(request_rows),
            "frozen_dates_in_scope": sum(
                1 for day, split in split_by_day.items()
                if "20260601" <= day <= "20260714" and split in {"train", "val", "validation"}
            ),
        },
        "window_contract": {
            "frames_per_window": WINDOW_FRAMES,
            "cadence_minutes": CADENCE_MINUTES,
            "same_BJT_day_only": True,
            "daytime_mask_applied": False,
            "label_or_pixel_payloads_read": False,
        },
        "by_split": result_by_split,
        "affected_days": affected_days,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
