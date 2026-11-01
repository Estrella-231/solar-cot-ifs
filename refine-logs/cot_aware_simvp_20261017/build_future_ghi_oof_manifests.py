#!/usr/bin/env python3
"""Build date-grouped, frame-disjoint OOF SimVP manifests from official train rows.

The official monthly manifest defines the upstream training population. The
forecast-bank index defines which rows later need OOF predictions. No payloads
or labels are opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bjt_day_from_path(relpath: str) -> str:
    stamp = Path(relpath).stem
    return (datetime.strptime(stamp, "%Y%m%d%H%M") + timedelta(hours=8)).date().isoformat()


def assign_folds(days: set[str], k: int, salt: str) -> dict[str, int]:
    if k < 2:
        raise ValueError("k must be >= 2")
    months: dict[str, list[str]] = defaultdict(list)
    for day in days:
        months[day[:7]].append(day)
    out: dict[str, int] = {}
    for month, month_days in months.items():
        ordered = sorted(month_days, key=lambda d: hashlib.sha256(f"{salt}:{k}:{d}".encode()).hexdigest())
        for i, day in enumerate(ordered):
            out[day] = i % k
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(official_manifest: Path, bank_index: Path, outdir: Path, k: int = 5,
          salt: str = "future_ghi_oof_20261019") -> dict:
    official = read_csv(official_manifest)
    official_train = [r for r in official if r.get("split") == "train"]
    by_id = {r["seq_id"]: r for r in official_train}
    if len(by_id) != len(official_train) or not by_id:
        raise ValueError("official train sequence IDs are empty or duplicated")
    bank_rows = [r for r in read_csv(bank_index) if r.get("split") == "train"]
    bank_ids = [r["seq_id"] for r in bank_rows]
    if not bank_ids or len(set(bank_ids)) != len(bank_ids):
        raise ValueError("bank train IDs are empty or duplicated")
    missing = sorted(set(bank_ids) - set(by_id))
    if missing:
        raise ValueError(f"bank train IDs missing from official train manifest: {missing[:5]}")

    days: set[str] = set()
    row_day: dict[str, str] = {}
    row_frames: dict[str, list[str]] = {}
    boundary_ids: set[str] = set()
    for seq_id, row in by_id.items():
        frames = row["data_relpaths"].split("|")
        if len(frames) != 24 or len(set(frames)) != 24:
            raise ValueError(f"expected 24 unique frames for {seq_id}")
        day = datetime.strptime(row["BJT_start"], "%Y-%m-%d %H:%M:%S").date().isoformat()
        path_days = {bjt_day_from_path(p) for p in frames}
        if day not in path_days:
            raise ValueError(f"manifest BJT_start does not match AGRI frames: {seq_id} {day} {sorted(path_days)}")
        if len(path_days) != 1:
            boundary_ids.add(seq_id)
            continue
        days.add(day)
        row_day[seq_id] = day
        row_frames[seq_id] = frames

    fold_for_day = assign_folds(days, k, salt)
    bank_set = set(bank_ids)
    if bank_set & boundary_ids:
        raise ValueError(f"bank requires OOF predictions for cross-date sequences: {sorted(bank_set & boundary_ids)[:5]}")
    summaries = []
    output_rows: dict[int, list[dict[str, str]]] = {}
    coverage: list[str] = []
    for fold in range(k):
        train_ids = [sid for sid in row_day if fold_for_day[row_day[sid]] != fold]
        held_ids = [sid for sid in row_day if fold_for_day[row_day[sid]] == fold]
        train_frames = {p for sid in train_ids for p in row_frames[sid]}
        held_frames = {p for sid in held_ids for p in row_frames[sid]}
        overlap = train_frames & held_frames
        if overlap:
            raise ValueError(f"fold {fold} has {len(overlap)} shared image frames")
        fold_oof = sorted(bank_set.intersection(held_ids))
        coverage.extend(fold_oof)
        train_rows = [{**by_id[sid], "split": "train"} for sid in train_ids]
        held_rows = [{**by_id[sid], "split": "val"} for sid in held_ids]
        manifest_rows = train_rows + held_rows
        output_rows[fold] = manifest_rows
        summaries.append({
            "fold": fold,
            "train_sequences": len(train_ids),
            "heldout_sequences": len(held_ids),
            "oof_bank_sequences": len(fold_oof),
            "train_unique_frames": len(train_frames),
            "heldout_unique_frames": len(held_frames),
            "train_heldout_frame_intersection": 0,
            "train_dates": len({row_day[s] for s in train_ids}),
            "heldout_dates": len({row_day[s] for s in held_ids}),
        })
    if sorted(coverage) != sorted(bank_ids) or len(coverage) != len(set(coverage)):
        raise ValueError("OOF prediction coverage is incomplete or duplicated")

    # Stage everything in a new directory; never overwrite existing evidence.
    if outdir.exists():
        raise FileExistsError(outdir)
    outdir.mkdir(parents=True)
    fields = ["split", "seq_id", "BJT_start", "data_relpaths"]
    for fold, rows in output_rows.items():
        d = outdir / f"fold_{fold}"
        d.mkdir()
        manifest = d / "hunan_8to16.csv"
        write_csv(manifest, rows, fields)
        train_paths = sorted({p for r in rows if r["split"] == "train" for p in r["data_relpaths"].split("|")})
        held_paths = sorted({p for r in rows if r["split"] == "val" for p in r["data_relpaths"].split("|")})
        (d / "train_frames.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
        (d / "val_frames.txt").write_text("\n".join(held_paths) + "\n", encoding="utf-8")
        (d / "test_frames.txt").write_text("", encoding="utf-8")
        audit = {
            "split_id": f"future_ghi_oof_5fold_{salt}_fold{fold}",
            "fold": fold,
            "k": k,
            "fold_salt": salt,
            "manifest_sha256": sha(manifest),
            "source_manifest_sha256": sha(official_manifest),
            "forecast_bank_index_sha256": sha(bank_index),
            "frame_disjoint_splits": True,
            "official_train_sequences": len(official_train),
            "eligible_train_sequences": len(row_day),
            "purged_cross_bjt_date_sequences": len(boundary_ids),
            "frame_list_sha256": {
                "train": sha(d / "train_frames.txt"),
                "val": sha(d / "val_frames.txt"),
                "test": sha(d / "test_frames.txt"),
            },
            "sequence_counts": {"train": sum(r["split"] == "train" for r in rows),
                                "val": sum(r["split"] == "val" for r in rows), "test": 0},
            "unique_frames": {"train": len(train_paths), "val": len(held_paths), "test": 0},
            "train_val_frame_intersection": 0,
            "training_population": "official-train rows whose 24 AGRI frames all fall within one BJT date; cross-date boundary sequences purged to prevent frame leakage",
            "heldout_oof_population": "heldout official-train dates; only forecast-bank train rows receive OOF predictions",
            "validation_test_used": False,
            "test_payloads_read": 0,
            "metadata_only": True,
        }
        (d / "manifest_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (d / "fold_oof_ids.txt").write_text("\n".join(sorted(bank_set.intersection(
            sid for sid in row_day if fold_for_day[row_day[sid]] == fold))) + "\n", encoding="utf-8")

    result = {
        "status": "PASS_OOF_MANIFESTS_CREATED",
        "k": k,
        "fold_salt": salt,
        "official_train_sequences": len(official_train),
        "eligible_official_train_sequences": len(row_day),
        "purged_cross_bjt_date_sequences": len(boundary_ids),
        "official_train_dates": len(days),
        "bank_train_sequences": len(bank_ids),
        "oof_bank_coverage": len(coverage),
        "folds": summaries,
        "official_manifest_sha256": sha(official_manifest),
        "bank_index_sha256": sha(bank_index),
        "metadata_only": True,
        "image_or_label_payload_opened": False,
        "validation_test_payload_opened": False,
    }
    (outdir / "manifest_build_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official-manifest", type=Path, required=True)
    p.add_argument("--bank-index", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--salt", default="future_ghi_oof_20261019")
    a = p.parse_args()
    print(json.dumps(build(a.official_manifest, a.bank_index, a.outdir, a.folds, a.salt), indent=2))


if __name__ == "__main__":
    main()
