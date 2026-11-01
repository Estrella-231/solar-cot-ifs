#!/usr/bin/env python3
"""Metadata-only proof that each SimVP OOF fold excludes its held-out dates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import re
from pathlib import Path


STAMP_RE = re.compile(r"(?<!\d)(\d{12})(?=\.npy$)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def row_dates(value: str) -> set[str]:
    paths = [x for x in value.split("|") if x]
    if not paths:
        raise ValueError("empty data_paths row")
    dates = set()
    for name in paths:
        match = STAMP_RE.search(Path(name).name)
        if match is None:
            raise ValueError(f"cannot parse timestamp from basename: {Path(name).name}")
        stamp = match.group(1)
        utc = datetime.strptime(stamp, "%Y%m%d%H%M")
        bjt = utc + timedelta(hours=8)
        dates.add(bjt.strftime("%Y-%m-%d"))
    if len(dates) != 1:
        raise ValueError(f"sequence crosses BJT dates: {sorted(dates)}")
    return dates


def read_index(path: Path) -> dict:
    digest = sha256(path)
    by_split: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "split" not in reader.fieldnames or "data_paths" not in reader.fieldnames:
            raise ValueError(f"unexpected manifest schema: {path}")
        for row in reader:
            split = row["split"]
            row_counts[split] = row_counts.get(split, 0) + 1
            by_split.setdefault(split, set()).update(row_dates(row["data_paths"]))
    return {"sha256": digest, "rows_by_split": row_counts,
            "dates_by_split": {k: sorted(v) for k, v in sorted(by_split.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", type=Path, required=True)
    ap.add_argument("--day-assignment", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    assignments: dict[str, dict] = {}
    with args.assignment.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            assignments[row["date_BJT"]] = row
    global_splits: dict[str, str] = {}
    with args.day_assignment.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            global_splits[row["date_BJT"]] = row["split"]
    result = {"state": "HUBEI_SIMVP_OOF_DATE_EXCLUSION_AUDIT_PASS_TEST_CLOSED",
              "assignment_sha256": sha256(args.assignment),
              "global_day_assignment_sha256": sha256(args.day_assignment), "filename_time_basis": "UTC filename converted to BJT by +08:00",
              "folds": {},
              "test_payloads_read": 0, "buffer_payloads_read": 0,
              "payload_arrays_read": 0, "check_scope": "CSV path/split metadata only"}
    for fold in range(5):
        fold_dir = args.fold_root / f"fold_{fold}"
        trainval_path = fold_dir / "simvp_trainval_index.csv"
        oof_path = fold_dir / "simvp_oof_index.csv"
        trainval = read_index(trainval_path)
        oof = read_index(oof_path)
        heldout = {d for d, row in assignments.items()
                   if row["split"] == "train" and int(row["heldout_fold"]) == fold}
        global_train = {d for d, split in global_splits.items() if split == "train"}
        global_val = {d for d, split in global_splits.items() if split == "val"}
        train_dates = set(trainval["dates_by_split"].get("train", []))
        oof_dates = set().union(*(set(v) for v in oof["dates_by_split"].values()))
        leaks = sorted(train_dates & heldout)
        unexpected_train = sorted(train_dates - global_train)
        unexpected_val = sorted(set(trainval["dates_by_split"].get("val", [])) - global_val)
        unexpected_oof = sorted(oof_dates - heldout)
        unexpected_splits = sorted(set(trainval["rows_by_split"]) - {"train", "val"})
        fold_state = "PASS" if not (leaks or unexpected_train or unexpected_val or unexpected_oof or unexpected_splits) and bool(oof_dates) else "FAIL"
        result["folds"][str(fold)] = {
            "state": fold_state, "heldout_assignment_dates": len(heldout),
            "simvp_train_dates": len(train_dates), "simvp_val_dates": len(trainval["dates_by_split"].get("val", [])),
            "simvp_oof_dates": len(oof_dates), "train_heldout_overlap": leaks,
            "train_dates_outside_global_train": unexpected_train,
            "val_dates_outside_global_val": unexpected_val,
            "oof_dates_outside_heldout": unexpected_oof,
            "unexpected_splits": unexpected_splits,
            "trainval_manifest_sha256": trainval["sha256"],
            "oof_manifest_sha256": oof["sha256"],
            "rows_by_split": trainval["rows_by_split"],
            "oof_rows_by_split": oof["rows_by_split"],
        }
    if any(v["state"] != "PASS" for v in result["folds"].values()):
        result["state"] = "HUBEI_SIMVP_OOF_DATE_EXCLUSION_AUDIT_FAIL_TEST_CLOSED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "folds": {
        k: {"state": v["state"], "overlap_count": len(v["train_heldout_overlap"]),
            "unexpected_oof_count": len(v["oof_dates_outside_heldout"])}
        for k, v in result["folds"].items()},
        "test_payloads_read": 0, "buffer_payloads_read": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
