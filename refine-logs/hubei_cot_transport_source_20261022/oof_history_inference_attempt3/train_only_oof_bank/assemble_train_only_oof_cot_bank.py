#!/usr/bin/env python3
"""Assemble the five date-held-out R shards into a train-only COT bank.

The full-bank CSV is consulted for split/key/order metadata only. No full-bank
COT array or validation COT value is opened. Every prediction value written to
the output comes from the matching date-held-out fold shard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-rows", type=Path, required=True,
                    help="Full train/val bank rows.csv; metadata only")
    ap.add_argument("--oof-root", type=Path, required=True,
                    help="Directory containing fold_0 ... fold_4 train-only shards")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out_dir}")

    bank_rows = read_rows(args.bank_rows)
    if len(bank_rows) != 22061:
        raise ValueError(f"unexpected bank metadata rows: {len(bank_rows)}")
    bank_index: dict[str, int] = {}
    train_rows: list[dict[str, str]] = []
    split_counts = {"train": 0, "val": 0}
    for row in bank_rows:
        key = row["time_bjt"]
        split = row["split"].strip().lower()
        if split not in split_counts or key in bank_index:
            raise ValueError(f"invalid split or duplicate key in bank metadata: {key}/{split}")
        bank_index[key] = int(row["index"])
        split_counts[split] += 1
        if split == "train":
            train_rows.append(row)
    if split_counts != {"train": 20206, "val": 1855}:
        raise ValueError(f"unexpected frozen split counts: {split_counts}")
    train_rows.sort(key=lambda r: int(r["index"]))
    train_keys = {r["time_bjt"] for r in train_rows}

    fold_sources: list[dict[str, str | int]] = []
    shard_by_key: dict[str, tuple[int, int]] = {}
    fold_data: list[tuple[np.ndarray, np.ndarray]] = []
    fold_rows: list[list[dict[str, str]]] = []
    for fold in range(5):
        d = args.oof_root / f"fold_{fold}"
        rows_path = d / "rows.csv"
        cot_path = d / "cot_oof.npy"
        day_path = d / "day_mask.npy"
        rows = read_rows(rows_path)
        cot = np.load(cot_path, mmap_mode="r", allow_pickle=False)
        day = np.load(day_path, mmap_mode="r", allow_pickle=False)
        if cot.shape != (len(rows), 16, 16) or cot.dtype != np.float32:
            raise ValueError(f"unexpected fold {fold} COT shape/dtype: {cot.shape}/{cot.dtype}")
        if day.shape != cot.shape or day.dtype != np.bool_:
            raise ValueError(f"unexpected fold {fold} day mask shape/dtype: {day.shape}/{day.dtype}")
        for i, row in enumerate(rows):
            key = row["time_bjt"]
            if (key not in train_keys or row["split"].strip().lower() != "train"
                    or int(row["heldout_fold"]) != fold or key in shard_by_key):
                raise ValueError(f"invalid or duplicate OOF train row: fold={fold}, key={key}")
            shard_by_key[key] = (fold, i)
        fold_data.append((cot, day))
        fold_rows.append(rows)
        for path in (rows_path, cot_path, day_path):
            fold_sources.append({"fold": fold, "file": str(path), "sha256": sha256(path)})

    if set(shard_by_key) != train_keys:
        missing = sorted(train_keys - set(shard_by_key))[:5]
        extra = sorted(set(shard_by_key) - train_keys)[:5]
        raise ValueError(f"OOF shard key coverage mismatch; missing={missing}, extra={extra}")

    n = len(train_rows)
    cot_out = np.empty((n, 1, 16, 16), dtype=np.float32)
    day_out = np.empty((n, 1, 16, 16), dtype=np.bool_)
    fold_out = np.empty(n, dtype=np.int8)
    key_to_out = {row["time_bjt"]: i for i, row in enumerate(train_rows)}
    for fold, rows in enumerate(fold_rows):
        cot_src, day_src = fold_data[fold]
        for src_i, row in enumerate(rows):
            out_i = key_to_out[row["time_bjt"]]
            cot_out[out_i, 0] = cot_src[src_i]
            day_out[out_i, 0] = day_src[src_i]
            fold_out[out_i] = fold

    if not np.isfinite(cot_out).all():
        raise ValueError("OOF train bank contains non-finite COT predictions")
    outside = (cot_out < 0) | (cot_out > 100)
    out_parent = args.out_dir.parent
    out_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=args.out_dir.name + ".tmp_", dir=out_parent))
    try:
        bank_path = tmp_dir / "cot_pred_train_oof.npz"
        np.savez_compressed(
            bank_path,
            cot=cot_out,
            day_mask=day_out,
            time_bjt=np.asarray([r["time_bjt"] for r in train_rows], dtype="U12"),
            heldout_fold=fold_out,
        )
        rows_path = tmp_dir / "rows_train.csv"
        with rows_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "time_bjt", "split", "heldout_fold"])
            for i, row in enumerate(train_rows):
                w.writerow([i, row["time_bjt"], "train", int(fold_out[i])])

        manifest = {
            "state": "COMPLETE_TRAIN_ONLY_OOF_R_COT_BANK",
            "scope": "Hubei train-only 16x16 R-COT predictions from date-held-out folds",
            "n_train_frames": n,
            "n_validation_metadata_rows_seen": split_counts["val"],
            "validation_prediction_values_read": 0,
            "test_payloads_read": 0,
            "buffer_payloads_read": 0,
            "cpp_target_or_qc_payloads_read": 0,
            "full_r_bank_prediction_values_read": 0,
            "split_counts_from_metadata": split_counts,
            "shape_cot": list(cot_out.shape),
            "dtype_cot": str(cot_out.dtype),
            "shape_day_mask": list(day_out.shape),
            "dtype_day_mask": str(day_out.dtype),
            "source_bank_rows_csv": str(args.bank_rows),
            "source_bank_rows_sha256": sha256(args.bank_rows),
            "fold_sources": fold_sources,
            "fold_frame_counts": {str(f): len(fold_rows[f]) for f in range(5)},
            "heldout_fold_counts": {str(f): int((fold_out == f).sum()) for f in range(5)},
            "cot_min_raw": float(cot_out.min()),
            "cot_max_raw": float(cot_out.max()),
            "cot_values_outside_0_100_preserved_not_clipped": int(outside.sum()),
            "history_validity_note": "Apply the existing day/SZA/finite/[0,100] validity rule downstream; raw OOF values are preserved here.",
            "outputs": {
                "cot_pred_train_oof.npz": sha256(bank_path),
                "rows_train.csv": sha256(rows_path),
            },
        }
        (tmp_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Refuse races/overwrites and publish the complete package atomically.
        if args.out_dir.exists():
            raise FileExistsError(f"refusing to overwrite output created concurrently: {args.out_dir}")
        os.rename(tmp_dir, args.out_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(json.dumps({
        "state": manifest["state"],
        "train_frames": n,
        "fold_counts": manifest["heldout_fold_counts"],
        "shape": manifest["shape_cot"],
        "cot_range": [manifest["cot_min_raw"], manifest["cot_max_raw"]],
        "outside_0_100_preserved": manifest["cot_values_outside_0_100_preserved_not_clipped"],
        "validation_values_read": 0,
        "test_buffer_cpp_values_read": 0,
        "output": str(args.out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
