#!/usr/bin/env python3
"""Integrity audit for one frozen-Hubei SimVP fold-out-of-fold bank.

Reads only the generated AGRI forecast bank and its train-only OOF index.
It does not read test/buffer payloads or any GHI labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

EXPECTED_CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
EXPECTED_LEAD_MINUTES = list(range(15, 241, 15))
EXPECTED_PATCH_ORIGIN = [103, 118]
EXPECTED_PATCH_SIZE = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_in_chunks(array: np.ndarray, rows_per_chunk: int = 32) -> bool:
    for start in range(0, array.shape[0], rows_per_chunk):
        if not np.isfinite(array[start:start + rows_per_chunk]).all():
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-count", type=int, default=797)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit receipt: {a.output}")

    report_path = a.bank / "BANK_REPORT.json"
    keys_path = a.bank / "val" / "keys.csv"
    pred_path = a.bank / "val" / "pred_agri_patch.npy"
    true_path = a.bank / "val" / "true_agri_patch.npy"
    geom_path = a.bank / "val" / "future_geometry_patch.npy"
    for path in (report_path, keys_path, pred_path, true_path, geom_path, a.index):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    with a.index.open("r", newline="", encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    with keys_path.open("r", newline="", encoding="utf-8") as f:
        key_rows = list(csv.DictReader(f))

    if report.get("state") != "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK":
        raise ValueError(f"Unexpected bank state: {report.get('state')}")
    if report.get("test_payloads_read") != 0:
        raise ValueError("Bank report does not certify test_payloads_read=0")
    if (report.get("buffer_payloads_read") != 0 or report.get("test_or_buffer_rows") != 0
            or report.get("channel_indices") != EXPECTED_CHANNEL_INDICES
            or report.get("lead_minutes") != EXPECTED_LEAD_MINUTES
            or report.get("patch_origin_crop_256") != EXPECTED_PATCH_ORIGIN
            or report.get("patch_size") != EXPECTED_PATCH_SIZE):
        raise ValueError("Bank report does not certify frozen shape/lead/test-buffer contract")
    split_counts = report.get("split_counts", {})
    if split_counts.get("test", 0) or split_counts.get("buffer", 0):
        raise ValueError(f"Index contains test/buffer rows: {split_counts}")
    if any(row.get("split") != "val" for row in index_rows):
        raise ValueError("OOF source index must contain only the fold's held-out rows")
    if len(index_rows) != a.expected_count or len(key_rows) != a.expected_count:
        raise ValueError(
            f"Count mismatch: index={len(index_rows)}, keys={len(key_rows)}, "
            f"expected={a.expected_count}"
        )
    if report.get("sequences_written_by_split", {}).get("val") != a.expected_count:
        raise ValueError("Report does not show the complete OOF row count")
    if report.get("index_sha256") != sha256(a.index):
        raise ValueError("Index SHA-256 differs from bank report")

    expected_shape = (a.expected_count, 16, 13, 16, 16)
    arrays = {
        "pred_agri_patch": np.load(pred_path, mmap_mode="r", allow_pickle=False),
        "true_agri_patch": np.load(true_path, mmap_mode="r", allow_pickle=False),
        "future_geometry_patch": np.load(geom_path, mmap_mode="r", allow_pickle=False),
    }
    if arrays["pred_agri_patch"].shape != expected_shape:
        raise ValueError(f"Prediction shape mismatch: {arrays['pred_agri_patch'].shape}")
    if arrays["true_agri_patch"].shape != expected_shape:
        raise ValueError(f"Truth shape mismatch: {arrays['true_agri_patch'].shape}")
    if arrays["future_geometry_patch"].shape != (a.expected_count, 16, 3, 16, 16):
        raise ValueError(f"Geometry shape mismatch: {arrays['future_geometry_patch'].shape}")
    for name, array in arrays.items():
        if not finite_in_chunks(array):
            raise ValueError(f"Non-finite values found in {name}")

    for i, (source, key) in enumerate(zip(index_rows, key_rows)):
        paths = source["data_paths"].split("|")
        if len(paths) != 24:
            raise ValueError(f"OOF index row {i} does not contain 24 ordered frames")
        utc = [datetime.strptime(Path(item).stem[:12], '%Y%m%d%H%M') for item in paths]
        bjt = [datetime.strptime(Path(item).stem[:12], '%Y%m%d%H%M')
               for item in source['aux_paths'].split('|')]
        if len(bjt) != 24 or any(g-u != timedelta(hours=8) for u,g in zip(utc,bjt)):
            raise ValueError('UTC source and BJT auxiliary timestamps are misaligned')
        stems = [t.strftime('%Y%m%d%H%M') for t in bjt]
        if int(key["sample_index"]) != i or key["split"] != "val":
            raise ValueError(f"Non-canonical or mislabeled bank key at row {i}")
        if key["init_time_bjt"] != stems[7]:
            raise ValueError(f"Initialization key mismatch at row {i}")
        if key["target_times_bjt"] != "|".join(stems[8:24]):
            raise ValueError(f"Target-time key mismatch at row {i}")
        if key["data_paths"] != source["data_paths"]:
            raise ValueError(f"Source key mismatch at row {i}")

    audit = {
        "state": "HUBEI_SIMVP_SINGLE_FOLD_OOF_BANK_INTEGRITY_PASS_TEST_CLOSED",
        "count": a.expected_count,
        "source_index_sha256": sha256(a.index),
        "source_bank_report_sha256": sha256(report_path),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "normalization_sha256": report.get("normalization_sha256"),
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "channel_indices": report.get("channel_indices"),
        "lead_minutes": report.get("lead_minutes"),
        "patch_origin_crop_256": report.get("patch_origin_crop_256"),
        "patch_size": report.get("patch_size"),
        "arrays_sha256": {
            "pred_agri_patch": sha256(pred_path),
            "true_agri_patch": sha256(true_path),
            "future_geometry_patch": sha256(geom_path),
        },
        "finite": True,
        "keys_exactly_match_ordered_oof_index": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "ghi_labels_read": False,
        "scope": "one-fold train-date OOF AGRI bank integrity only; not a GHI skill result",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
