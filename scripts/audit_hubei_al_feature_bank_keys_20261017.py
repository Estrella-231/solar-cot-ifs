#!/usr/bin/env python3
"""Audit the AL frozen-R feature bank against the frozen train/val GHI keys."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tkey(value: str) -> str:
    return value.strip().replace("-", "").replace(":", "").replace(" ", "")[:12]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--feature-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    pack_report_path = a.pack / "pack_report.json"
    pack_meta = json.loads(pack_report_path.read_text(encoding="utf-8"))
    feature_report_path = a.feature_bank / "AL_FEATURE_REPORT.json"
    feature_meta = json.loads(feature_report_path.read_text(encoding="utf-8"))
    assert pack_meta["state"] == "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED"
    assert pack_meta["test_and_buffer_values_read"] == 0
    assert feature_meta["state"] == "HUBEI_FROZEN_R_DEC1_AL_FEATURE_BANK_TRAINVAL_COMPLETE_TEST_CLOSED"
    assert feature_meta["pack_report_sha256"] == sha(pack_report_path)
    assert feature_meta["split_sha256"] == pack_meta["split_sha256"]
    assert feature_meta["test_payloads_read"] == 0
    assert feature_meta["ghi_labels_read"] is False
    assert feature_meta["cpp_payloads_read"] == 0

    ghi_rows = read_csv(a.pack / "rows.csv")
    feature_rows = read_csv(a.feature_bank / "rows.csv")
    assert len(ghi_rows) == pack_meta["samples"] == 29351
    ghi_keys = {(tkey(r["time_bjt"]), r["split"]) for r in ghi_rows}
    station_keys = {(r["station"], tkey(r["time_bjt"]), r["split"]) for r in ghi_rows}
    feature_keys = {(tkey(r["time_bjt"]), r["split"]) for r in feature_rows}
    assert len(station_keys) == len(ghi_rows)
    assert len(ghi_keys) == 12301
    assert len(feature_rows) == len(feature_keys) == len(ghi_keys)
    assert feature_keys == ghi_keys, "feature bank keys differ from GHI train/val time-split keys"
    assert all(r["split"] in ("train", "val") for r in ghi_rows + feature_rows)

    array_path = a.feature_bank / "r_dec1_station_3x3.npy"
    arr = np.load(array_path, mmap_mode="r", allow_pickle=False)
    assert arr.shape == (len(feature_rows), 3, 16, 3, 3)
    assert np.isfinite(arr).all()
    assert feature_meta["feature_file_sha256"] == sha(array_path)
    assert feature_meta["rows_sha256"] == sha(a.feature_bank / "rows.csv")

    result = {
        "state": "HUBEI_AL_FEATURE_BANK_KEY_AUDIT_PASS_TEST_CLOSED",
        "pack_report_sha256": sha(pack_report_path),
        "pack_split_sha256": pack_meta["split_sha256"],
        "ghi_station_rows": len(ghi_rows),
        "unique_time_split_keys": len(ghi_keys),
        "feature_rows": len(feature_rows),
        "feature_shape": list(arr.shape),
        "key_sets_equal": True,
        "all_values_finite": True,
        "test_payloads_read": 0,
        "ghi_labels_read_by_feature_extractor": False,
        "cpp_payloads_read_by_feature_extractor": 0,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
