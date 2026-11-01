#!/usr/bin/env python3
"""Independent row/array/provenance audit for the common-support four-arm pack."""
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
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-ghi-pack", required=True, type=Path)
    p.add_argument("--cot-bank", required=True, type=Path)
    p.add_argument("--pcloud-bank", required=True, type=Path)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = json.loads((a.pack / "pack_report.json").read_text(encoding="utf-8"))
    if report.get("state") != "HUBEI_GHI_FOURARM_COMMON_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("unexpected pack state")
    if report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError("test/buffer boundary failed")
    source_rows = read_rows(a.source_ghi_pack / "rows.csv")
    rows = read_rows(a.pack / "rows.csv")
    cot_rows, pc_rows = read_rows(a.cot_bank / "rows.csv"), read_rows(a.pcloud_bank / "rows.csv")
    cot_map = {r["time_bjt"]: i for i, r in enumerate(cot_rows)}
    pc_map = {r["time_bjt"]: i for i, r in enumerate(pc_rows)}
    if len(cot_map) != len(cot_rows) or len(pc_map) != len(pc_rows):
        raise ValueError("duplicate auxiliary time key")
    source_index = {}
    for i, r in enumerate(source_rows):
        k = (r["station"], r["time_bjt"], r["split"])
        if k in source_index:
            raise ValueError(f"duplicate source row {k}")
        source_index[k] = i
    expected = [r for r in source_rows if r["time_bjt"] in cot_map and r["time_bjt"] in pc_map]
    if [(r["station"], r["time_bjt"], r["split"]) for r in rows] != [
        (r["station"], r["time_bjt"], r["split"]) for r in expected
    ]:
        raise ValueError("pack is not the ordered exact common support")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("non-train/val row in pack")
    keep = np.asarray([source_index[(r["station"], r["time_bjt"], r["split"])] for r in rows], dtype=np.int64)
    for name in BASE_FILES:
        src = np.load(a.source_ghi_pack / name, mmap_mode="r", allow_pickle=False)
        got = np.load(a.pack / name, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(got, src[keep]):
            raise ValueError(f"source GHI values changed after common-support join: {name}")

    cot = np.load(a.cot_bank / "cot_pred_trainval_full.npy", mmap_mode="r", allow_pickle=False)
    pc = np.load(a.pcloud_bank / "p_cloud_trainval.npy", mmap_mode="r", allow_pickle=False)
    got_cot = np.load(a.pack / "cot_pred_3x3.npy", mmap_mode="r", allow_pickle=False)
    got_pc = np.load(a.pack / "p_cloud_3x3.npy", mmap_mode="r", allow_pickle=False)
    if got_cot.shape != (len(rows), 1, 3, 3) or got_pc.shape != (len(rows), 1, 3, 3):
        raise ValueError("auxiliary 3x3 shape mismatch")
    source_count = {"date_oof": 0, "full_train": 0}
    for i, r in enumerate(rows):
        ci, pi = cot_map[r["time_bjt"]], pc_map[r["time_bjt"]]
        if cot_rows[ci]["split"] != r["split"] or pc_rows[pi]["split"] != r["split"]:
            raise ValueError(f"split mismatch at {r['time_bjt']}")
        rr, cc = CENTERS[r["station"]]
        sl = np.s_[rr - 1:rr + 2, cc - 1:cc + 2]
        if not np.array_equal(got_cot[i, 0], np.asarray(cot[ci, 0][sl], dtype=np.float32)):
            raise ValueError(f"COT crop/key mismatch at {r['station']}/{r['time_bjt']}")
        if not np.array_equal(got_pc[i, 0], np.asarray(pc[pi][sl], dtype=np.float32)):
            raise ValueError(f"p_cloud crop/key mismatch at {r['station']}/{r['time_bjt']}")
        src = pc_rows[pi].get("prediction_source", "")
        if src not in source_count:
            raise ValueError(f"unknown classifier source {src}")
        source_count[src] += 1
    if source_count != report.get("pcloud_source_rows_used"):
        raise ValueError("classifier OOF/full-train source counts differ from pack report")
    out = {
        "state": "PASS_HUBEI_FOURARM_PACK_INDEPENDENT_COMMON_SUPPORT_AUDIT_TEST_CLOSED",
        "source_ghi_rows": len(source_rows), "retained_rows": len(rows),
        "excluded_coverage_rows": len(source_rows) - len(rows),
        "rows_by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
        "rows_by_station": {s: sum(r["station"] == s for r in rows) for s in CENTERS},
        "source_arrays_exact_equal_after_key_ordered_join": True,
        "cot_and_pcloud_key_split_crop_alignment": "PASS",
        "pcloud_prediction_sources": source_count,
        "pack_report_sha256": sha(a.pack / "pack_report.json"),
        "test_payloads_read": 0, "buffer_payloads_read": 0,
    }
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
