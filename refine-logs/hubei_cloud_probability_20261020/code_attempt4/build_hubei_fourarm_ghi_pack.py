#!/usr/bin/env python3
"""Build the four-arm Hubei GHI pack with corrected COT and p_cloud banks.

Reads only the audited same-time train/validation GHI arrays and row ledger
from the old pack. It deliberately never opens the old, time-misaligned cot.npy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
BASE_FILES = ("agri_raw.npy", "geometry.npy", "ghi_wm2.npy", "clear_sky_wm2.npy")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError(f"Expected train/val-only rows: {path}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ghi-pack", required=True, type=Path)
    p.add_argument("--cot-bank", required=True, type=Path)
    p.add_argument("--pcloud-bank", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--exclude-all-zero-agri", action="store_true",
                    help="apply the shared input-quality rule: exclude a station row if all 13 AGRI bands are zero throughout its 3x3 input patch")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    old_report_path = a.ghi_pack / "pack_report.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    if old_report.get("state") != "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("unexpected source GHI pack state")
    if old_report.get("test_payloads_read", old_report.get("test_and_buffer_values_read")) != 0:
        raise ValueError("source GHI pack is not test-closed")
    rows = read_rows(a.ghi_pack / "rows.csv")
    source_agri = np.load(a.ghi_pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    if source_agri.shape != (len(rows), 13, 3, 3):
        raise ValueError(f"unexpected source AGRI shape {source_agri.shape}")
    cot_report = json.loads((a.cot_bank / "BANK_REPORT.json").read_text(encoding="utf-8"))
    pc_report = json.loads((a.pcloud_bank / "BANK_REPORT.json").read_text(encoding="utf-8"))
    if cot_report.get("state") != "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED":
        raise ValueError("corrected Hubei train/val R COT bank required")
    if pc_report.get("state") != "COMPLETE_HUBEI_FULL_AGRI_P_CLOUD_TRAINVAL_TEST_CLOSED":
        raise ValueError("corrected Hubei p_cloud train/val bank required")
    if cot_report.get("test_payloads_read") != 0 or pc_report.get("test_payloads_read") != 0:
        raise ValueError("sidecar bank test boundary failure")

    cot_rows = read_rows(a.cot_bank / "rows.csv")
    pc_rows = read_rows(a.pcloud_bank / "rows.csv")
    if len(cot_rows) != len(pc_rows):
        raise ValueError("COT and p_cloud row counts differ")
    cot_by_key = {r["time_bjt"]: (i, r) for i, r in enumerate(cot_rows)}
    pc_by_key = {r["time_bjt"]: (i, r) for i, r in enumerate(pc_rows)}
    if len(cot_by_key) != len(cot_rows) or len(pc_by_key) != len(pc_rows):
        raise ValueError("duplicate time key in a sidecar bank")
    cot = np.load(a.cot_bank / "cot_pred_trainval_full.npy", mmap_mode="r", allow_pickle=False)
    pcloud = np.load(a.pcloud_bank / "p_cloud_trainval.npy", mmap_mode="r", allow_pickle=False)
    if cot.shape != (len(cot_rows), 1, 16, 16) or pcloud.shape != (len(pc_rows), 16, 16):
        raise ValueError("unexpected full-bank shapes")

    # The source AGRI manifest does not cover every row in the historical GHI
    # pack. Compare all four arms on the strict common source support; never
    # silently fill absent COT/cloud features or select rows by weather/result.
    keep_idx, kept_rows = [], []
    excluded = {st: {sp: 0 for sp in ("train", "val")} for st in CENTERS}
    excluded_keys = {sp: set() for sp in ("train", "val")}
    invalid_zero_agri = []
    invalid_by_station_split = {st: {sp: 0 for sp in ("train", "val")} for st in CENTERS}
    for i, row in enumerate(rows):
        key, st, split = row["time_bjt"], row["station"], row["split"]
        if st not in CENTERS:
            raise ValueError(f"unexpected station {st}")
        has_cot, has_pc = key in cot_by_key, key in pc_by_key
        if not (has_cot and has_pc):
            excluded[st][split] += 1
            excluded_keys[split].add(key)
            continue
        if cot_by_key[key][1]["split"] != split or pc_by_key[key][1]["split"] != split:
            raise ValueError(f"split mismatch at {st}/{key}")
        if a.exclude_all_zero_agri and not np.any(source_agri[i] != 0):
            invalid_zero_agri.append({"source_row_index": i, "station": st, "split": split,
                                      "time_bjt": key, "reason": "all_13_agri_bands_zero_in_station_3x3"})
            invalid_by_station_split[st][split] += 1
            continue
        keep_idx.append(i)
        kept_rows.append(row)
    if not keep_idx:
        raise ValueError("no common train/validation rows across GHI, COT and p_cloud")
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    n = len(kept_rows)
    cot_3x3 = np.empty((n, 1, 3, 3), dtype=np.float32)
    pcloud_3x3 = np.empty((n, 1, 3, 3), dtype=np.float32)
    source_counts = {s: 0 for s in ("date_oof", "full_train")}
    for out_i, row in enumerate(kept_rows):
        key, st, split = row["time_bjt"], row["station"], row["split"]
        ci, cr = cot_by_key[key]
        pi, pr = pc_by_key[key]
        rr, cc = CENTERS[st]
        sl = np.s_[rr - 1:rr + 2, cc - 1:cc + 2]
        cot_3x3[out_i, 0] = np.asarray(cot[ci, 0][sl], dtype=np.float32)
        pcloud_3x3[out_i, 0] = np.asarray(pcloud[pi][sl], dtype=np.float32)
        src = pr.get("prediction_source", "")
        if src not in source_counts:
            raise ValueError(f"unexpected p_cloud source {src!r}")
        source_counts[src] += 1
    if not np.isfinite(cot_3x3).all() or np.any(cot_3x3 < 0):
        raise ValueError("invalid corrected COT feature")
    if not np.isfinite(pcloud_3x3).all() or np.any((pcloud_3x3 < 0) | (pcloud_3x3 > 1)):
        raise ValueError("invalid cloud probability feature")

    a.output.mkdir(parents=True, exist_ok=False)
    base_hashes = {}
    for name in BASE_FILES:
        src = a.ghi_pack / name
        if not src.is_file():
            raise FileNotFoundError(src)
        source_array = source_agri if name == "agri_raw.npy" else np.load(src, mmap_mode="r", allow_pickle=False)
        if source_array.shape[0] != len(rows):
            raise ValueError(f"source array/row count mismatch: {name}")
        retained = np.asarray(source_array[keep_idx])
        np.save(a.output / name, retained, allow_pickle=False)
        # Exact rowwise comparison after the coverage-only common-support join.
        if not np.array_equal(np.load(a.output / name, allow_pickle=False), retained):
            raise RuntimeError(f"retained source rows changed while writing: {name}")
        base_hashes[name] = {
            "source_sha256": sha(src),
            "retained_array_sha256": sha(a.output / name),
            "rowwise_equal_to_source_common_support": True,
            "source_shape": list(source_array.shape),
            "retained_shape": list(retained.shape),
        }
    with (a.output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept_rows)
    np.save(a.output / "cot_pred_3x3.npy", cot_3x3, allow_pickle=False)
    np.save(a.output / "p_cloud_3x3.npy", pcloud_3x3, allow_pickle=False)
    hashes = {name: sha(a.output / name) for name in (*BASE_FILES, "rows.csv", "cot_pred_3x3.npy", "p_cloud_3x3.npy")}
    out_report = {
        "state": "HUBEI_GHI_FOURARM_COMMON_TRAINVAL_PACK_TEST_CLOSED",
        "rows": n,
        "source_ghi_rows": len(rows),
        "retained_rows_after_coverage_and_quality": n,
        "auxiliary_common_support_rows_before_quality_filter": n + len(invalid_zero_agri),
        "excluded_rows_due_aux_source_unavailable": sum(v for by_split in excluded.values() for v in by_split.values()),
        "total_rows_excluded_from_fourarm_pack": len(rows) - n,
        "input_quality_exclusion_rule": ("none" if not a.exclude_all_zero_agri else
            "exclude station rows where all 13 AGRI bands are exactly zero across the station-centred 3x3 input; applied identically before all four arms; raw source pack unchanged"),
        "excluded_rows_all_zero_agri_station_3x3": len(invalid_zero_agri),
        "excluded_all_zero_agri_by_station_split": invalid_by_station_split,
        "excluded_all_zero_agri_keys": invalid_zero_agri,
        "excluded_rows_by_station_split": excluded,
        "unique_missing_time_keys_by_split": {s: len(v) for s, v in excluded_keys.items()},
        "exclusion_policy": "coverage-only common support across existing GHI rows, corrected COT bank and p_cloud bank; identical retained rows for all arms; no weather/performance filtering",
        "rows_by_split": {s: int(sum(r["split"] == s for r in kept_rows)) for s in ("train", "val")},
        "rows_by_station": {st: int(sum(r["station"] == st for r in kept_rows)) for st in CENTERS},
        "arms": ["A", "AC", "AM", "ACM"],
        "input_geometry": "station-centered 3x3 AGRI+cosSOZ+cosRAA; same-time GHI reconstruction",
        "stations_local_centers_16x16": CENTERS,
        "base_arrays_exactly_equal_to_source_on_retained_row_indices": base_hashes,
        "source_row_indices_sha256": hashlib.sha256(keep_idx.astype("<i8").tobytes()).hexdigest(),
        "source_ghi_pack_report_sha256": sha(old_report_path),
        "source_rows_sha256": sha(a.ghi_pack / "rows.csv"),
        "cot_bank_report_sha256": sha(a.cot_bank / "BANK_REPORT.json"),
        "cot_bank_prediction_sha256": cot_report.get("prediction_sha256"),
        "pcloud_bank_report_sha256": sha(a.pcloud_bank / "BANK_REPORT.json"),
        "pcloud_bank_prediction_sha256": pc_report.get("prediction_sha256"),
        "pcloud_source_rows_used": source_counts,
        "old_time_misaligned_cot_array_read": False,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "hashes": hashes,
        "interpretation": "CPP-derived COT and CLP-trained cloud probability are predicted features; CPP CLP/COT are retrieval references, not independent truth",
    }
    (a.output / "pack_report.json").write_text(json.dumps(out_report, indent=2, ensure_ascii=False), encoding="utf-8")
    (a.output / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(out_report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
