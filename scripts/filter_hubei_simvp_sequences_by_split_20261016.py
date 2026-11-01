#!/usr/bin/env python3
"""Filter SimVP sequences to strict 24-frame single-split date support.

Metadata-only operation: it reads sequence paths and the frozen day assignment,
never opens AGRI/geometry payloads, GHI labels, CPP/COT, test, or buffer data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SOURCE_INDEX_SHA = "5ccdc3d99d0e6c3838701952dc3c407943f10efd178aa42b407ec297f8215a43"
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_days(path: Path) -> dict[str, str]:
    days: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            day = row["date_BJT"].replace("-", "")
            if day in days:
                raise ValueError(f"Duplicate date assignment: {day}")
            days[day] = row["split"]
    return days


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--day-assignment", type=Path, required=True)
    p.add_argument("--output-index", type=Path, required=True)
    p.add_argument("--output-report", type=Path, required=True)
    a = p.parse_args()
    if a.output_index.exists() or a.output_report.exists():
        raise FileExistsError("Refusing to overwrite strict-index outputs")
    source_sha, split_sha = sha256(a.source_index), sha256(a.day_assignment)
    if source_sha != EXPECTED_SOURCE_INDEX_SHA:
        raise ValueError(f"Unexpected source index SHA-256: {source_sha}")
    if split_sha != EXPECTED_SPLIT_SHA:
        raise ValueError(f"Unexpected frozen day-assignment SHA-256: {split_sha}")
    day_split = read_days(a.day_assignment)
    retained: list[dict[str, str]] = []
    rejected_by_split: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    checked = 0
    fields: list[str] | None = None
    with a.source_index.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if fields != ["split", "data_paths", "aux_paths"]:
            raise ValueError(f"Unexpected index schema: {fields}")
        for row in reader:
            split = row["split"]
            if split not in ("train", "val"):
                raise ValueError(f"Non train/validation sequence in source index: {split}")
            data_paths = row["data_paths"].split("|")
            aux_paths = row["aux_paths"].split("|")
            if len(data_paths) != 24 or len(aux_paths) != 24:
                raise ValueError("Every SimVP sequence must contain 24 aligned frame/geometry paths")
            checked += 1
            dates = []
            for path in data_paths:
                stamp = Path(path).stem
                if len(stamp) != 12 or not stamp.isdigit():
                    raise ValueError(f"Invalid BJT frame timestamp in index: {path}")
                dates.append(stamp[:8])
            statuses = [day_split.get(day) for day in dates]
            cross_bjt_day = len(set(dates)) != 1
            if cross_bjt_day:
                rejected_reasons[f"{split}:cross_bjt_day"] += 1
            split_mismatch = not all(status == split for status in statuses)
            if split_mismatch:
                rejected_reasons[f"{split}:split_mismatch"] += 1
            if not cross_bjt_day and not split_mismatch:
                retained.append(row)
                continue
            rejected_by_split[split] += 1
            if any(status is None for status in statuses):
                rejected_reasons[f"{split}:unassigned_calendar_day"] += 1
            if any(status == "buffer" for status in statuses):
                rejected_reasons[f"{split}:buffer_frame"] += 1
            if any(status == "test" for status in statuses):
                rejected_reasons[f"{split}:test_frame"] += 1
            if any(status not in (split, "buffer", "test", None) for status in statuses):
                rejected_reasons[f"{split}:other_split_frame"] += 1
    a.output_index.parent.mkdir(parents=True, exist_ok=True)
    with a.output_index.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)
    report = {
        "state": "STRICT_24_FRAME_SPLIT_INDEX_METADATA_AUDIT_PASS_TEST_BUFFER_CLOSED",
        "source_index": str(a.source_index),
        "source_index_sha256": source_sha,
        "day_assignment": str(a.day_assignment),
        "day_assignment_sha256": split_sha,
        "output_index": str(a.output_index),
        "output_index_sha256": sha256(a.output_index),
        "rows_checked": checked,
        "rows_by_split_before": {s: sum(1 for r in retained if r["split"] == s) + rejected_by_split[s] for s in ("train", "val")},
        "rows_by_split_after": {s: sum(1 for r in retained if r["split"] == s) for s in ("train", "val")},
        "rejected_by_split": dict(rejected_by_split),
        "rejected_reasons_nonexclusive": dict(rejected_reasons),
        "rule": "all 24 BJT scene timestamps (8 history + 16 targets) must belong to one BJT calendar day, and that day must map to the row split in the frozen day assignment",
        "sequence_path_metadata_only": True,
        "agri_or_aux_payloads_read": 0,
        "ghi_values_read": 0,
        "test_or_buffer_payloads_read": 0,
        "test_or_buffer_payloads_in_output_index": 0,
    }
    a.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
