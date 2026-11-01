#!/usr/bin/env python3
"""Bind the forecast-head train/val GHI arrays to the frozen 15-min label CSV.

Reads only train/validation pack rows and label tables. It does not inspect
test/buffer values or modify any model/result artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "trainval_labels_20261015_v1"
RECEIPT_DIR = ROOT / "audit_receipts" / "forecast_ghi_label_binding_20261017"
LABEL_CSV = LABEL_DIR / "ghi15_train_val.csv"
DAYLIGHT_CLEAR_CSV = LABEL_DIR / "ghi15_train_val_daylight_clear.csv"
LABEL_REPORT = LABEL_DIR / "REPORT.json"
PACK_REPORT = RECEIPT_DIR / "pack_report.json"
ROWS_CSV = RECEIPT_DIR / "rows.csv"
GHI_NPY = RECEIPT_DIR / "ghi_wm2.npy"
CLEAR_NPY = RECEIPT_DIR / "clear_sky_wm2.npy"
OUT = RECEIPT_DIR / "forecast_ghi_label_binding_audit_v3.json"
DAYLIGHT_REPORT = LABEL_DIR / "DAYLIGHT_REPORT.json"
CLEAR_REPORT = LABEL_DIR / "CLEAR_SKY_REPORT.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    label_report = json.loads(LABEL_REPORT.read_text(encoding="utf-8"))
    pack_report = json.loads(PACK_REPORT.read_text(encoding="utf-8"))
    if label_report.get("test_radiation_values_read") != 0 or label_report.get("test_radiation_cells_decoded") != 0:
        raise ValueError("Source label report does not certify closed test values")
    if label_report.get("window") != "(target-15min,target], 15 right-closed minute bins; minute means equally weighted":
        raise ValueError("Unexpected source-label window contract")
    if pack_report.get("state") != "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED":
        raise ValueError("Forecast pack is not train/validation only")
    if pack_report.get("test_and_buffer_values_read") != 0:
        raise ValueError("Forecast pack did not certify closed test/buffer")

    raw_label_rows = records(LABEL_CSV)
    clear_label_rows = records(DAYLIGHT_CLEAR_CSV)
    labels: dict[tuple[str, str], tuple[str, float, float]] = {}
    clear_labels: dict[tuple[str, str], tuple[float, float]] = {}
    clear_splits: dict[tuple[str, str], str] = {}
    for row in raw_label_rows:
        if row.get("split") not in {"train", "val"}:
            raise ValueError("Raw label table contains a non-train/validation split")
        if row.get("mask") == "1":
            key = (row["station"], row["target_time_BJT"].replace("-", "").replace(":", "").replace(" ", "")[:12])
            labels[key] = (row["split"], float(row["ghi15_raw"]), float("nan"))
    for row in clear_label_rows:
        if row.get("split") not in {"train", "val"}:
            raise ValueError("Clear-sky label table contains a non-train/validation split")
        if row.get("valid_daylight_clear_sky_label") not in {"1", "1.0", "True", "true"}:
            continue
        key = (row["station"], row["target_time_BJT"].replace("-", "").replace(":", "").replace(" ", "")[:12])
        clear_labels[key] = (float(row["ghi15_raw"]), float(row["clear_sky_ghi_15min_wm2"]))
        clear_splits[key] = row["split"]

    # Independently bind the derived daylight/clear table back to the raw
    # 15-minute source-label table before comparing either with the pack.
    source_table_bad = []
    source_table_max_abs = 0.0
    for key, (derived_ghi, _clear_sky) in clear_labels.items():
        if key not in labels:
            source_table_bad.append({"key": key, "error": "derived row missing from raw label table"})
            continue
        raw_split, raw_ghi, _ = labels[key]
        if raw_split != clear_splits[key]:
            source_table_bad.append({"key": key, "error": "raw/derived split mismatch"})
        delta = abs(raw_ghi - derived_ghi)
        source_table_max_abs = max(source_table_max_abs, delta)
        if delta > 2e-4:
            source_table_bad.append({"key": key, "error": "raw/derived GHI mismatch", "abs_wm2": delta})

    pack_rows = records(ROWS_CSV)
    ghi = np.load(GHI_NPY, allow_pickle=False)
    clear = np.load(CLEAR_NPY, allow_pickle=False)
    if ghi.shape != (len(pack_rows),) or clear.shape != (len(pack_rows),):
        raise ValueError("Array lengths differ from the forecast pack rows")
    bad = []
    seen = set()
    max_ghi_abs = 0.0
    max_clear_abs = 0.0
    for i, row in enumerate(pack_rows):
        if row["split"] not in {"train", "val"}:
            raise ValueError("Forecast pack contains a non-train/validation row")
        station = row["station"]
        target = row["target_time_bjt"]
        key = (station, target)
        if key in seen:
            # A target can legitimately occur for multiple forecast init times;
            # this audit therefore tracks the full key for duplicate checking below.
            pass
        full_key = (row["split"], station, row["init_time_bjt"], target, row["lead_minutes"])
        if full_key in seen:
            raise ValueError(f"Duplicate forecast pack key: {full_key}")
        seen.add(full_key)
        lookup = (station, target)
        if lookup not in clear_labels or lookup not in labels:
            bad.append({"row": i, "key": full_key, "error": "missing matching valid source label"})
            continue
        raw_split, raw_ghi, _ = labels[lookup]
        expected_ghi, expected_clear = clear_labels[lookup]
        if raw_split != row["split"]:
            bad.append({"row": i, "key": full_key, "error": "split mismatch"})
            continue
        d_ghi = abs(float(ghi[i]) - expected_ghi)
        d_clear = abs(float(clear[i]) - expected_clear)
        max_ghi_abs = max(max_ghi_abs, d_ghi)
        max_clear_abs = max(max_clear_abs, d_clear)
        if d_ghi > 2e-4 or d_clear > 2e-4:
            bad.append({"row": i, "key": full_key, "error": "array differs from source label", "ghi_abs": d_ghi, "clear_abs": d_clear})
        if abs(float(row["ghi_wm2"]) - expected_ghi) > 2e-4 or abs(float(row["clear_sky_wm2"]) - expected_clear) > 2e-4:
            bad.append({"row": i, "key": full_key, "error": "rows.csv differs from source label"})

    report = {
        "status": "HUBEI_FORECAST_GHI_LABEL_BINDING_PASS_TEST_CLOSED" if not bad and not source_table_bad else "FAIL",
        "scope": "train_validation_only; no test or buffer payloads read",
        "pack_samples": len(pack_rows),
        "matched_source_label_rows": len(pack_rows) - len(bad),
        "raw_to_daylight_source_rows_checked": len(clear_labels),
        "raw_to_daylight_source_mismatch_count": len(source_table_bad),
        "max_abs_raw_15min_ghi_vs_daylight_table_wm2": source_table_max_abs,
        "mismatch_count": len(bad),
        "max_abs_ghi_vs_15min_source_label_wm2": max_ghi_abs,
        "max_abs_clear_sky_vs_source_label_wm2": max_clear_abs,
        "source_window": label_report["window"],
        "source_field": label_report["source_field"],
        "source_unit_contract": label_report["unit_contract"],
        "source_label_sha256": label_report["label_csv_sha256"],
        "source_label_path": str(LABEL_CSV.resolve()),
        "source_workbooks": label_report["source_files"],
        "source_workbook_hashes_are_copied_from_label_builder_report_not_reverified_in_this_step": True,
        "daylight_clear_label_sha256": sha(DAYLIGHT_CLEAR_CSV),
        "daylight_report_sha256": sha(DAYLIGHT_REPORT),
        "clear_sky_report_sha256": sha(CLEAR_REPORT),
        "pack_rows_sha256": sha(ROWS_CSV),
        "pack_ghi_array_sha256": sha(GHI_NPY),
        "pack_clear_sky_array_sha256": sha(CLEAR_NPY),
        "pack_report_sha256": sha(PACK_REPORT),
        "label_builder_source_sha256": sha(Path("solar-cot-ifs/scripts/build_hubei_ghi15_trainval.py")),
        "pack_files_copied_read_only_from": "/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_forecast_ghi_20261017/pack/",
        "sample_errors": bad[:20],
        "raw_to_daylight_source_errors": source_table_bad[:20],
        "test_values_read": 0,
        "buffer_values_read": 0,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if bad or source_table_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
