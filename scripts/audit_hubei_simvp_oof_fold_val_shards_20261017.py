#!/usr/bin/env python3
"""Audit a checkpoint-reloaded, validation-only SimVP forecast bank.

This verifier is intentionally restricted to one frozen fold's validation
split. It checks shard lineage, exact canonical key coverage, tensor shape and
finite values, while keeping test/buffer payloads closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PRED_TAIL = (16, 13, 16, 16)
GEOM_TAIL = (16, 3, 16, 16)
CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
LEAD_MINUTES = list(range(15, 241, 15))
PATCH_ORIGIN_CROP_256 = [103, 118]
PATCH_SIZE = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_key(row: dict[str, str], sample_index: int) -> dict[str, str]:
    paths = row["data_paths"].split("|")
    if len(paths) != 24:
        raise RuntimeError(f"Expected 24 data paths at validation row {sample_index}")
    stamps = [Path(p).stem for p in paths]
    return {
        "sample_index": str(sample_index),
        "split": "val",
        "init_time_bjt": stamps[7],
        "target_times_bjt": "|".join(stamps[8:24]),
        "data_paths": row["data_paths"],
    }


def assert_finite(path: Path, expected_shape: tuple[int, ...]) -> None:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.shape != expected_shape:
        raise RuntimeError(f"BAD_SHAPE {path}: {arr.shape} != {expected_shape}")
    step = max(1, min(32, arr.shape[0]))
    for start in range(0, arr.shape[0], step):
        if not np.isfinite(arr[start:start + step]).all():
            raise RuntimeError(f"NONFINITE_VALUES {path} rows={start}:{start + step}")


def validate_source_bank_report(report: dict, index_sha: str, shard_count: int) -> int:
    """Fail closed on split, tensor semantics, and explicit test/buffer counters."""
    split_counts = report.get("split_counts", {})
    sequences_by_split = report.get("sequences_by_split", {})
    written_by_split = report.get("sequences_written_by_split", {})
    val_count = int(written_by_split.get("val", -1))
    if (report.get("state") != "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK"
            or report.get("test_payloads_read") != 0
            or report.get("buffer_payloads_read") != 0
            or report.get("test_or_buffer_rows") != 0
            or report.get("index_sha256") != index_sha
            or report.get("inferred_splits") != ["val"]
            or report.get("shard_count") != shard_count
            or report.get("channel_indices") != CHANNEL_INDICES
            or report.get("lead_minutes") != LEAD_MINUTES
            or report.get("patch_origin_crop_256") != PATCH_ORIGIN_CROP_256
            or report.get("patch_size") != PATCH_SIZE
            or split_counts.get("test") != 0
            or split_counts.get("buffer") != 0
            or sequences_by_split != {"val": val_count}
            or written_by_split.get("train") != 0
            or written_by_split.get("val") != val_count
            or val_count < 0):
        raise RuntimeError("Validation shard source report violates frozen tensor/split/read contract")
    return val_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True,
                    help="Frozen fold train/validation index; only val payloads are read")
    ap.add_argument("--fold", type=int, required=True, choices=range(5),
                    help="Frozen fold ID whose checkpoint is being reloaded")
    ap.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True,
                    help="New JSON audit output path")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    index_sha = sha256(args.index)
    train_n = val_n = 0
    val_rows: list[dict[str, str]] = []
    with args.index.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["split", "data_paths", "aux_paths"]:
            raise RuntimeError(f"Unexpected index columns: {reader.fieldnames}")
        for row in reader:
            if row["split"] == "train":
                train_n += 1
            elif row["split"] == "val":
                val_rows.append(row)
                val_n += 1
            else:
                raise RuntimeError(f"Refusing non-train/val index row: {row['split']!r}")
    if val_n == 0:
        raise RuntimeError("No frozen validation rows")

    reports = []
    all_keys: dict[int, dict[str, str]] = {}
    weighted_mse = weighted_mae = 0.0
    total_elements = total_sequences = 0
    common = None
    for shard_root in args.shard_dir:
        rp = shard_root / "BANK_REPORT.json"
        if not rp.is_file():
            raise FileNotFoundError(rp)
        report = json.loads(rp.read_text(encoding="utf-8"))
        val_count = validate_source_bank_report(report, index_sha, len(args.shard_dir))
        local_stage = report.get("local_stage")
        if local_stage is not None and not isinstance(local_stage, dict):
            raise RuntimeError("local_stage must be null or an object")
        stage_sha = (local_stage or {}).get("stage_manifest_sha256")
        if local_stage is not None and not stage_sha:
            raise RuntimeError("A declared local stage must bind its manifest SHA-256")
        identity = (report.get("checkpoint_sha256"), report.get("normalization_sha256"),
                    report.get("index_sha256"), stage_sha)
        if (not all(identity[:3])
                or (common is not None and identity != common)):
            raise RuntimeError("Shards do not share checkpoint/norm/index/stage lineage")
        common = identity

        split_report = report.get("full_field_validation_metrics", {}).get("val")
        if not isinstance(split_report, dict):
            raise RuntimeError(f"Missing full-field val metrics: {rp}")
        n_elem = int(split_report.get("element_count", 0))
        n_seq = int(split_report.get("sequence_count", -1))
        if n_elem <= 0 or n_seq != val_count:
            raise RuntimeError(f"Bad metric counts in {rp}")
        mse = float(split_report["normalized_mse"])
        mae = float(split_report["normalized_mae"])
        if not np.isfinite([mse, mae]).all():
            raise RuntimeError(f"Non-finite validation metric in {rp}")
        weighted_mse += mse * n_elem
        weighted_mae += mae * n_elem
        total_elements += n_elem
        total_sequences += n_seq

        keys = read_csv(shard_root / "val" / "keys.csv")
        if len(keys) != val_count:
            raise RuntimeError(f"Key count differs from report in {shard_root}")
        for key in keys:
            i = int(key["sample_index"])
            if not 0 <= i < val_n or i in all_keys:
                raise RuntimeError(f"Out-of-range or duplicate val key index {i}")
            if key != expected_key(val_rows[i], i):
                raise RuntimeError(f"Canonical validation key mismatch at row {i}")
            all_keys[i] = key

        pred_path = shard_root / "val" / "pred_agri_patch.npy"
        true_path = shard_root / "val" / "true_agri_patch.npy"
        geom_path = shard_root / "val" / "future_geometry_patch.npy"
        assert_finite(pred_path, (val_count, *PRED_TAIL))
        assert_finite(true_path, (val_count, *PRED_TAIL))
        assert_finite(geom_path, (val_count, *GEOM_TAIL))
        reports.append({
            "shard_index": int(report["shard_index"]),
            "sequence_count": val_count,
            "local_stage_manifest_sha256": stage_sha,
            "channel_indices": report["channel_indices"],
            "lead_minutes": report["lead_minutes"],
            "patch_origin_crop_256": report["patch_origin_crop_256"],
            "patch_size": report["patch_size"],
            "test_payloads_read": report["test_payloads_read"],
            "buffer_payloads_read": report["buffer_payloads_read"],
            "test_or_buffer_rows": report["test_or_buffer_rows"],
            "prediction_sha256": sha256(pred_path),
            "truth_sha256": sha256(true_path),
            "geometry_sha256": sha256(geom_path),
            "keys_sha256": sha256(shard_root / "val" / "keys.csv"),
            "bank_report_sha256": sha256(rp),
        })

    shard_ids = sorted(x["shard_index"] for x in reports)
    expected_shard_ids = list(range(len(args.shard_dir)))
    if shard_ids != expected_shard_ids:
        raise RuntimeError(f"Incomplete/duplicate shard IDs: {shard_ids}")
    if sorted(all_keys) != list(range(val_n)) or total_sequences != val_n:
        raise RuntimeError(f"Incomplete validation coverage: keys={len(all_keys)} expected={val_n}")
    if total_elements <= 0:
        raise RuntimeError("No validation field elements were scored")

    result = {
        "status": f"HUBEI_SIMVP_FOLD{args.fold}_CHECKPOINT_RELOAD_VALIDATION_AUDIT_PASSED_TEST_CLOSED",
        "fold_id": args.fold,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "train_index_rows_metadata_only": train_n,
        "validation_sequences_inferred": val_n,
        "unique_validation_key_coverage": len(all_keys),
        "checkpoint_sha256": common[0],
        "normalization_sha256": common[1],
        "index_sha256": common[2],
        "stage_manifest_sha256": common[3],
        "channel_indices": CHANNEL_INDICES,
        "lead_minutes": LEAD_MINUTES,
        "patch_origin_crop_256": PATCH_ORIGIN_CROP_256,
        "patch_size": PATCH_SIZE,
        "shard_count": len(reports),
        "shard_ids": shard_ids,
        "full_field_validation_metrics": {
            "normalized_mse": weighted_mse / total_elements,
            "normalized_mae": weighted_mae / total_elements,
            "element_count": total_elements,
            "sequence_count": total_sequences,
            "scope": "all 13 predicted AGRI channels, all 16 leads and full 256x256 field in frozen normalized space",
        },
        "shards": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
