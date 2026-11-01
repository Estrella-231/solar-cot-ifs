#!/usr/bin/env python3
"""Extract frozen T16 validation cases for plotting after the blind manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def utc_key(iso_bjt: str) -> str:
    t = datetime.fromisoformat(iso_bjt) - timedelta(hours=8)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cpp-pack", type=Path, required=True)
    ap.add_argument("--ghi-pack", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--head-results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    event = json.loads(a.manifest.read_text(encoding="utf-8"))
    if event.get("state") != "PARTIAL_PRESELECTED_CPP_VALIDATION_CASES" or event.get("test_payloads_read") != 0:
        raise ValueError("T16 manifest is not frozen/preselected validation output")
    cases = event["selected_cases"]
    if len(cases) != 4 or event.get("ghi_values_accessed") is not False:
        raise ValueError("unexpected frozen case set")

    cpp_rows = list(csv.DictReader((a.cpp_pack / "rows.csv").open(newline="", encoding="utf-8")))
    cpp_cot = np.load(a.cpp_pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    cpp_mask = np.load(a.cpp_pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    x_raw = np.load(a.cpp_pack / "x_raw.npy", mmap_mode="r", allow_pickle=False)
    valid_row_by_index = {int(r["array_index"]): r for r in cpp_rows if r["state"] == "SUPERVISED"}
    ghi_pack_meta = json.loads((a.ghi_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if ghi_pack_meta.get("test_payload_opened") is not False:
        raise ValueError("GHI pack test gate failed")
    ghi_rows = list(csv.DictReader((a.ghi_pack / "rows.csv").open(newline="", encoding="utf-8")))
    val_rows = {"sili": [], "zhujia": []}
    for r in ghi_rows:
        if r["split"] == "val":
            val_rows[r["station"]].append(r)
    val_index = {
        s: {(r["station"], r["target_time_utc"]): i for i, r in enumerate(val_rows[s])}
        for s in val_rows
    }
    prediction_arrays = {}
    for seed in (42, 43, 44):
        for station in ("sili", "zhujia"):
            seed_dir = a.head_results / f"ghi_heads_new_r_seed{seed}_v1" / station
            for arm in ("A", "AC", "AP"):
                p = seed_dir / f"{arm}_val_prediction_ghi_wm2.npy"
                arr = np.asarray(np.load(p, allow_pickle=False), dtype=np.float64).reshape(-1)
                if len(arr) != len(val_rows[station]):
                    raise ValueError(f"prediction and validation key count differ: {seed}/{station}/{arm}")
                prediction_arrays[(seed, station, arm)] = arr
    labels = {}
    with a.labels.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["new_split"] == "val" and r["main_mask"].lower() == "true":
                labels[(r["station"], r["target_time_utc"])] = r

    rgb, cot, cot_mask, observed, clear, preds = [], [], [], [], [], []
    for case in cases:
        station = case["station"]
        ids = [int(i) for i in case["pack_array_indices"]]
        if len(ids) != 17:
            raise ValueError("case does not contain all 17 timestamps")
        # Channels in x_raw follow C01,C02,C03,C04,C05,C06,...; false-color is
        # R=C06, G=C03, B=C02. Values remain raw; display scaling is recorded later.
        rgb.append(np.asarray(x_raw[ids][:, [5, 2, 1]], dtype=np.float32))
        cot.append(np.asarray(cpp_cot[ids, 0], dtype=np.float32))
        cot_mask.append(np.asarray(cpp_mask[ids, 0], dtype=bool))
        case_obs, case_clear, case_pred = [], [], []
        for iso in case["timestamps_bjt"]:
            key = (station, utc_key(iso))
            if key not in labels or key not in val_index[station]:
                raise ValueError(f"frozen event lacks a paired GHI validation label/prediction: {key}")
            row = labels[key]
            case_obs.append(float(row["observed_ghi_15min_wm2"]))
            case_clear.append(float(row["clear_sky_ghi_15min_wm2"]))
            k = val_index[station][key]
            seed_pred = []
            for seed in (42, 43, 44):
                arm_pred = [float(prediction_arrays[(seed, station, arm)][k]) for arm in ("A", "AC", "AP")]
                seed_pred.append(arm_pred)
            case_obs, case_clear = case_obs, case_clear
            case_pred.append(seed_pred)
        observed.append(case_obs)
        clear.append(case_clear)
        preds.append(case_pred)

    arrays = {
        "false_color_rgb_raw": np.asarray(rgb, dtype=np.float32),
        "cpp_cot_physical": np.asarray(cot, dtype=np.float32),
        "cpp_quality_mask": np.asarray(cot_mask, dtype=np.uint8),
        "observed_ghi_15min_wm2": np.asarray(observed, dtype=np.float64),
        "clear_sky_ghi_15min_wm2": np.asarray(clear, dtype=np.float64),
        "ghi_head_predictions_3seed": np.asarray(preds, dtype=np.float64),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **arrays)
    meta = {
        "state": "T16_CASE_PAYLOAD_EXTRACTED",
        "event_manifest_sha256": sha256(a.manifest),
        "cpp_pack_audit_sha256": sha256(a.cpp_pack / "pack_audit.json"),
        "ghi_pack_manifest_sha256": sha256(a.ghi_pack / "pack_manifest.json"),
        "ghi_label_table_sha256": sha256(a.labels),
        "test_payloads_read": 0,
        "GHI_numeric_values_read_after_manifest_freeze": True,
        "GHI_prediction_inputs_are": "same-time observed AGRI, solar geometry and R-COT; diagnostic replay, not a forecast",
        "false_color_bands": {"red": "C06 (2.25 um)", "green": "C03 (0.86 um)", "blue": "C02 (0.65 um)"},
        "transforms": "raw AGRI reflectance values; case-global 2nd/98th percentile display normalization is applied by plotting script; no spatial interpolation",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "source": "train/validation-only CPP station16 pack, GHI pack, frozen validation labels and saved A/AC/AP predictions",
    }
    a.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
