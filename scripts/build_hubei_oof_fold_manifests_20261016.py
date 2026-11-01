#!/usr/bin/env python3
"""Build metadata-only, fold-isolated Hubei S/R train and OOF manifests.

No AGRI, geometry, CPP, GHI, test, or buffer payload is opened. For each fold,
the fixed global validation dates remain the only checkpoint-selection set;
the fold's global-train dates are emitted separately for held-out inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


EXPECTED_INDEX_SHA = "4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62"
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
EXPECTED_FOLDS_SHA = "c86bd6713d5afea1d6864cc4ef6048a93b596051ea4358e7954f39c32b054122"
EXPECTED_R_PATHS_SHA = "3bae20a9607b941e544199b9f2900414b266ad11456e82d9bf51e7bd78c972f8"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"Missing CSV header: {path}")
        return fields, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def date_for_stamp(stamp: str) -> str:
    if len(stamp) != 12 or not stamp.isdigit():
        raise ValueError(f"Expected YYYYMMDDHHMM timestamp, got {stamp!r}")
    datetime.strptime(stamp, "%Y%m%d%H%M")
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def sequence_date(row: dict[str, str]) -> str:
    data_paths = row["data_paths"].split("|")
    aux_paths = row["aux_paths"].split("|")
    if len(data_paths) != 24 or len(aux_paths) != 24:
        raise ValueError("Every sequence must contain 24 aligned data/geometry paths")
    dates = {date_for_stamp(Path(p).stem[:12]) for p in data_paths}
    if len(dates) != 1:
        raise ValueError(f"Sequence crosses BJT calendar dates: {sorted(dates)}")
    # Keep the exact ordered data/aux pairing from the frozen source index.
    # Their filenames use different time coordinates in this legacy pipeline;
    # do not infer alignment by comparing filename stamps here.
    return next(iter(dates))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--day-assignment", type=Path, required=True)
    p.add_argument("--fold-assignment", type=Path, required=True)
    p.add_argument("--r-path-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()

    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output_dir}")
    source_hash, split_hash, folds_hash = map(sha256,
        (a.source_index, a.day_assignment, a.fold_assignment))
    if source_hash != EXPECTED_INDEX_SHA:
        raise ValueError(f"Unexpected strict source index SHA-256: {source_hash}")
    if split_hash != EXPECTED_SPLIT_SHA:
        raise ValueError(f"Unexpected frozen day-assignment SHA-256: {split_hash}")
    if folds_hash != EXPECTED_FOLDS_SHA:
        raise ValueError(f"Unexpected OOF fold-manifest SHA-256: {folds_hash}")
    r_paths_hash = sha256(a.r_path_manifest)
    if EXPECTED_R_PATHS_SHA and r_paths_hash != EXPECTED_R_PATHS_SHA:
        raise ValueError(f"Unexpected R path manifest SHA-256: {r_paths_hash}")

    split_fields, split_rows = read_csv(a.day_assignment)
    if not {"date_BJT", "split"}.issubset(split_fields):
        raise ValueError("Unexpected day-assignment schema")
    split_by_date: dict[str, str] = {}
    for row in split_rows:
        day = row["date_BJT"]
        if day in split_by_date:
            raise ValueError(f"Duplicate frozen date: {day}")
        if row["split"] not in {"train", "val", "test", "buffer"}:
            raise ValueError(f"Unexpected split value: {row['split']}")
        split_by_date[day] = row["split"]

    fold_fields, fold_rows = read_csv(a.fold_assignment)
    if not {"date_BJT", "split", "heldout_fold"}.issubset(fold_fields):
        raise ValueError("Unexpected OOF fold schema")
    fold_by_date: dict[str, int] = {}
    for row in fold_rows:
        day = row["date_BJT"]
        if day in fold_by_date or row["split"] != "train":
            raise ValueError(f"Duplicate or non-train OOF date: {row}")
        if split_by_date.get(day) != "train":
            raise ValueError(f"OOF date is not in frozen global train: {day}")
        fold = int(row["heldout_fold"])
        if not 0 <= fold < 5:
            raise ValueError(f"Invalid fold number: {fold}")
        fold_by_date[day] = fold
    train_days = {d for d, s in split_by_date.items() if s == "train"}
    if set(fold_by_date) != train_days:
        raise ValueError("Fold dates do not exactly cover frozen train dates")

    index_fields, index_rows = read_csv(a.source_index)
    if index_fields != ["split", "data_paths", "aux_paths"]:
        raise ValueError(f"Unexpected strict index fields: {index_fields}")
    trainval_by_date: Counter[str] = Counter()
    for row in index_rows:
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"Source index unexpectedly contains {row['split']}")
        day = sequence_date(row)
        if split_by_date.get(day) != row["split"]:
            raise ValueError(f"Source index split/date mismatch: {row['split']} {day}")
        trainval_by_date[day] += 1
    source_train_days = {d for d in trainval_by_date if split_by_date[d] == "train"}
    source_val_days = {d for d in trainval_by_date if split_by_date[d] == "val"}
    if not source_train_days.issubset(train_days):
        raise ValueError("Strict S index contains dates absent from the frozen train-date fold set")

    r_fields, r_rows = read_csv(a.r_path_manifest)
    if not {"time_bjt", "split", "path_agri", "path_cpp_fixed_v2"}.issubset(r_fields):
        raise ValueError(f"Unexpected R manifest fields: {r_fields}")
    seen_r: set[str] = set()
    r_by_date: Counter[str] = Counter()
    for row in r_rows:
        stamp = row["time_bjt"]
        if stamp in seen_r:
            raise ValueError(f"Duplicate R time: {stamp}")
        seen_r.add(stamp)
        day = date_for_stamp(stamp)
        if split_by_date.get(day) != row["split"]:
            raise ValueError(f"R manifest split/date mismatch: {row['split']} {day}")
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"R manifest includes non-train/val row: {row['split']}")
        r_by_date[day] += 1
    source_r_train_days = {d for d in r_by_date if split_by_date[d] == "train"}
    source_r_val_days = {d for d in r_by_date if split_by_date[d] == "val"}

    a.output_dir.mkdir(parents=True, exist_ok=False)
    all_reports = []
    for fold in range(5):
        fold_dir = a.output_dir / f"fold_{fold}"
        fold_dir.mkdir()
        trainval_rows: list[dict[str, str]] = []
        oof_rows: list[dict[str, str]] = []
        for row in index_rows:
            day = sequence_date(row)
            if row["split"] == "train" and fold_by_date[day] == fold:
                oof_rows.append(dict(row, split="val"))
            else:
                trainval_rows.append(dict(row))

        r_fold_rows: list[dict[str, str]] = []
        for row in r_rows:
            day = date_for_stamp(row["time_bjt"])
            if row["split"] == "train" and fold_by_date[day] == fold:
                continue
            r_fold_rows.append(dict(row))

        if not oof_rows:
            raise ValueError(f"No heldout SimVP sequences for fold {fold}")
        if not any(r["split"] == "train" for r in trainval_rows):
            raise ValueError(f"No S training rows remain for fold {fold}")
        if not any(r["split"] == "train" for r in r_fold_rows):
            raise ValueError(f"No R training rows remain for fold {fold}")
        write_csv(fold_dir / "simvp_trainval_index.csv", index_fields, trainval_rows)
        write_csv(fold_dir / "simvp_oof_index.csv", index_fields, oof_rows)
        write_csv(fold_dir / "r_trainval_manifest.csv", r_fields, r_fold_rows)

        train_index_dates = {sequence_date(r) for r in trainval_rows if r["split"] == "train"}
        val_index_dates = {sequence_date(r) for r in trainval_rows if r["split"] == "val"}
        oof_dates = {sequence_date(r) for r in oof_rows}
        r_train_dates = {date_for_stamp(r["time_bjt"]) for r in r_fold_rows if r["split"] == "train"}
        r_val_dates = {date_for_stamp(r["time_bjt"]) for r in r_fold_rows if r["split"] == "val"}
        if train_index_dates & oof_dates or r_train_dates & oof_dates:
            raise RuntimeError(f"Fold {fold}: train/OOF date leakage")
        if val_index_dates != source_val_days or r_val_dates != source_r_val_days:
            raise RuntimeError(f"Fold {fold}: fixed global S/R validation support changed")
        expected_s_oof_dates = {d for d in source_train_days if fold_by_date[d] == fold}
        if train_index_dates | oof_dates != source_train_days or oof_dates != expected_s_oof_dates:
            raise RuntimeError(f"Fold {fold}: S train+OOF date coverage mismatch")
        expected_r_train_dates = source_r_train_days - {d for d, f in fold_by_date.items() if f == fold}
        if r_train_dates != expected_r_train_dates:
            raise RuntimeError(f"Fold {fold}: R train rows do not exactly exclude this fold")
        if r_val_dates != source_r_val_days:
            raise RuntimeError(f"Fold {fold}: R fixed validation support changed")

        report = {
            "fold": fold,
            "state": "METADATA_ONLY_FOLD_MANIFESTS_PASS_TEST_BUFFER_CLOSED",
            "simvp_trainval_index_sha256": sha256(fold_dir / "simvp_trainval_index.csv"),
            "simvp_oof_index_sha256": sha256(fold_dir / "simvp_oof_index.csv"),
            "r_trainval_manifest_sha256": sha256(fold_dir / "r_trainval_manifest.csv"),
            "simvp_sequences_by_split": dict(Counter(r["split"] for r in trainval_rows)),
            "simvp_oof_sequences": len(oof_rows),
            "r_rows_by_split": dict(Counter(r["split"] for r in r_fold_rows)),
            "simvp_train_dates": len(train_index_dates),
            "simvp_oof_dates": len(oof_dates),
            "simvp_train_dates_missing_from_strict_index": len(train_days - source_train_days),
            "fixed_validation_dates": len(val_index_dates),
            "r_train_oof_date_overlap": 0,
            "r_supervision_train_dates_missing_from_manifest": len(train_days - source_r_train_days),
            "test_buffer_payloads_read": 0,
            "payloads_read": 0,
            "checkpoint_selection_rule": "only the unchanged global validation dates; never select on this fold's OOF dates",
            "agri_aux_pairing": "preserved as ordered in the frozen source index; filenames are not assumed to share a time coordinate",
        }
        (fold_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_reports.append(report)

    root_report = {
        "state": "HUBEI_FIVE_FOLD_S_R_MANIFESTS_BUILT_METADATA_ONLY",
        "source_index_sha256": source_hash,
        "day_assignment_sha256": split_hash,
        "fold_assignment_sha256": folds_hash,
        "r_path_manifest_sha256": r_paths_hash,
        "fold_count": 5,
        "frozen_global_train_dates": len(train_days),
        "strict_s_train_dates_with_sequences": len(source_train_days),
        "global_train_dates_without_strict_s_sequences": len(train_days - source_train_days),
        "frozen_global_validation_dates": len(source_val_days),
        "s_trainval_sequences": dict(Counter(r["split"] for r in index_rows)),
        "r_manifest_rows": dict(Counter(r["split"] for r in r_rows)),
        "r_supervision_train_dates": len(source_r_train_days),
        "r_supervision_validation_dates": len(source_r_val_days),
        "r_supervision_train_dates_missing_from_s_train": len(train_days - source_r_train_days),
        "test_buffer_payloads_read": 0,
        "payloads_read": 0,
        "folds": all_reports,
        "next_gate": "fit fold-specific S/R normalization from each fold train only; execute payload-based audits before any training",
    }
    (a.output_dir / "root_report.json").write_text(json.dumps(root_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(root_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
