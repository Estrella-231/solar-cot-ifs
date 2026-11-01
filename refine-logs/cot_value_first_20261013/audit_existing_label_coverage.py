"""Count overlap of new train/val target keys with an existing label pack.

The candidate manifest contains train/val keys only. The old pack is scanned
for the two key columns; label values are not emitted or summarized here.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime


def norm_time(value: str) -> str:
    value = value.strip()
    d = datetime.fromisoformat(value)
    if d.tzinfo is not None:
        from zoneinfo import ZoneInfo
        d = d.astimezone(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
    return d.isoformat(sep=" ")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--existing-rows", required=True)
    a = p.parse_args()

    candidates: dict[tuple[str, str], str] = {}
    with open(a.candidates, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in {"train", "val"}:
                raise ValueError(f"unexpected candidate split: {split}")
            key = row["station"], norm_time(row["target_time_bjt"])
            if key in candidates:
                raise ValueError(f"duplicate candidate key: {key}")
            candidates[key] = split

    matched_source_splits: dict[tuple[str, str], set[str]] = {}
    repeats: Counter[tuple[str, str]] = Counter()
    with open(a.existing_rows, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"station", "target_time_bjt", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"existing rows lack columns: {sorted(required-set(reader.fieldnames or []))}")
        for row in reader:
            key = row["station"], norm_time(row["target_time_bjt"])
            if key not in candidates:
                continue
            matched_source_splits.setdefault(key, set()).add(row["split"])
            repeats[key] += 1

    counts = Counter()
    source_split_counts = Counter()
    conflicting_source_split_keys = 0
    repeated_keys = 0
    for key, split in candidates.items():
        counts[(split, "candidate")] += 1
        if key not in matched_source_splits:
            counts[(split, "uncovered_by_existing_pack")] += 1
        else:
            counts[(split, "covered_by_existing_pack")] += 1
            if repeats[key] > 1:
                repeated_keys += 1
            if len(matched_source_splits[key]) > 1:
                conflicting_source_split_keys += 1
            for old_split in matched_source_splits[key]:
                source_split_counts[(split, old_split)] += 1

    result = {
        "state": "EXISTING_LABEL_KEY_COVERAGE_AUDITED",
        "candidate_keys": len(candidates),
        "counts": {"/".join(k): v for k, v in sorted(counts.items())},
        "candidate_keys_repeated_in_existing_rows": repeated_keys,
        "keys_with_conflicting_old_split": conflicting_source_split_keys,
        "matched_candidate_keys_by_old_split": {
            f"{new}|{old}": n for (new, old), n in sorted(source_split_counts.items())
        },
        "label_values_summarized": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
