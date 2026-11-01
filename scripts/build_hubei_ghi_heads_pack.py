#!/usr/bin/env python3
"""Join frozen Hubei train/val labels to real AGRI and frozen-R 3x3 features.

All A/AC/AP arms consume exactly the same rows. Test/buffer label values are
rejected; only rows with a complete 15-min label, daylight geometry, positive
interval clear-sky irradiance, and an available AGRI/COT feature are materialized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

AGRI_CH = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_time(value: str) -> str:
    text = value.strip().replace("-", "").replace(":", "").replace(" ", "")
    if len(text) == 12:
        return text
    if len(text) == 14:
        return text[:12]
    raise ValueError(f"Unsupported BJT time key: {value!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--feature-rows", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--feature-report", type=Path, required=True)
    p.add_argument("--r-bank-report", type=Path, required=True)
    p.add_argument("--r-prediction", type=Path, required=True)
    p.add_argument("--r-rows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.split) != SPLIT_SHA:
        raise ValueError("Frozen Hubei split checksum mismatch")

    feature_report = json.loads(a.feature_report.read_text(encoding="utf-8"))
    r_report = json.loads(a.r_bank_report.read_text(encoding="utf-8"))
    if feature_report.get("state") != "HUBEI_STATION_3X3_RAW_FEATURES_TRAINVAL_COMPLETE_TEST_CLOSED":
        raise ValueError("Audited frozen-R feature report required")
    if r_report.get("state") != "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED":
        raise ValueError("Audited frozen-R bank report required")
    if feature_report.get("test_payloads_read") != 0 or r_report.get("test_payloads_read") != 0:
        raise ValueError("Test payload boundary violation")
    input_hashes = {
        "features": sha(a.features), "feature_rows": sha(a.feature_rows),
        "feature_report": sha(a.feature_report), "r_bank_report": sha(a.r_bank_report),
        "r_prediction": sha(a.r_prediction), "r_rows": sha(a.r_rows),
    }
    if feature_report.get("features_sha256") != input_hashes["features"]:
        raise ValueError("Feature report does not bind the supplied feature array")
    if feature_report.get("rows_sha256") != input_hashes["feature_rows"]:
        raise ValueError("Feature report does not bind the supplied feature rows")
    if feature_report.get("r_bank_report_sha256") != input_hashes["r_bank_report"]:
        raise ValueError("Feature report does not bind the supplied R bank report")
    if feature_report.get("r_bank_prediction_sha256") != input_hashes["r_prediction"]:
        raise ValueError("Feature report does not bind the supplied R prediction array")
    if feature_report.get("r_bank_rows_sha256") != input_hashes["r_rows"]:
        raise ValueError("Feature report does not bind the supplied R bank rows")
    if r_report.get("prediction_sha256") != input_hashes["r_prediction"]:
        raise ValueError("R bank report does not bind the supplied R prediction array")
    if r_report.get("rows_sha256") != input_hashes["r_rows"]:
        raise ValueError("R bank report does not bind the supplied R bank rows")
    if feature_report.get("split_sha256") != SPLIT_SHA:
        raise ValueError("Feature report uses a different frozen split")
    if feature_report.get("r_manifest_sha256") != r_report.get("manifest_sha256"):
        raise ValueError("Feature report and R bank report have different source manifests")

    # Time/split key map is metadata only and must not include test values.
    with a.split.open(encoding="utf-8-sig", newline="") as f:
        split_by_date = {r["date_BJT"].replace("-", ""): r["split"] for r in csv.DictReader(f)}

    feature_rows = list(csv.DictReader(a.feature_rows.open(encoding="utf-8-sig", newline="")))
    r_rows = list(csv.DictReader(a.r_rows.open(encoding="utf-8-sig", newline="")))
    if len(feature_rows) != len(r_rows):
        raise ValueError("Feature rows and R bank rows have different lengths")
    for i, (fr, rr) in enumerate(zip(feature_rows, r_rows)):
        if any(str(fr.get(k, "")) != str(rr.get(k, "")) for k in ("index", "time_bjt", "split", "path_agri")):
            raise ValueError(f"Feature rows are not aligned to R bank rows at index {i}")
    all_features = np.load(a.features, mmap_mode="r", allow_pickle=False)
    if all_features.shape != (len(feature_rows), 3, 17, 3, 3):
        raise ValueError(f"Feature bank shape mismatch: {all_features.shape}")
    feature_by_time: dict[str, tuple[str, np.ndarray]] = {}
    for i, r in enumerate(feature_rows):
        t = normalize_time(r["time_bjt"])
        if t in feature_by_time:
            raise ValueError(f"Duplicate old feature timestamp {t}")
        if r["split"] not in ("train", "val"):
            raise ValueError("Old feature rows contain a non-train/val member")
        feature_by_time[t] = (r["split"], np.asarray(all_features[i], dtype=np.float32))

    for t, (feature_split, station_features) in feature_by_time.items():
        date = t[:8]
        expected_split = split_by_date.get(date)
        if expected_split not in ("train", "val") or feature_split != expected_split:
            raise ValueError(f"Feature row has non-train/val or wrong split at {t}")
        if not np.isfinite(station_features).all():
            raise ValueError(f"Nonfinite feature row at {t}")

    samples = []
    seen = set()
    with a.labels.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["station"] not in STATIONS:
                continue
            split = r["split"]
            if split not in ("train", "val"):
                # Split metadata may cover all frozen dates. Skip held-out
                # rows before touching any radiation/clear-sky value fields.
                continue
            t = normalize_time(r["target_time_BJT"])
            if split_by_date.get(t[:8]) != split:
                raise ValueError(f"Label/split map mismatch at {r['station']} {t}")
            if r["valid_daylight_clear_sky_label"] not in ("1", "1.0", "True", "true"):
                continue
            if not r["ghi15_raw"]:
                raise ValueError(f"Masked label unexpectedly selected at {r['station']} {t}")
            if t not in feature_by_time:
                continue
            feature_split, station_features = feature_by_time[t]
            if feature_split != split:
                raise ValueError(f"Feature/label split mismatch at {r['station']} {t}")
            key = (r["station"], t)
            if key in seen:
                raise ValueError(f"Duplicate common sample key {key}")
            seen.add(key)
            si = STATIONS.index(r["station"])
            item = station_features[si]
            samples.append({
                "split": split, "station": r["station"], "time_bjt": t,
                "agri": item[:13], "geometry": item[13:15], "cot": item[16:17],
                "ghi": float(r["ghi15_raw"]),
                "clear": float(r["clear_sky_ghi_15min_wm2"]),
            })

    if not samples:
        raise RuntimeError("No common train/validation samples")
    if any(not np.isfinite(s[k]).all() for s in samples for k in ("agri", "geometry", "cot")):
        raise ValueError("Nonfinite model inputs")
    if any(not np.isfinite(s[k]) for s in samples for k in ("ghi", "clear")):
        raise ValueError("Nonfinite labels/clear sky")

    a.output.mkdir(parents=True)
    n = len(samples)
    arrays = {
        "agri_raw.npy": np.stack([s["agri"] for s in samples]).astype(np.float32),
        "geometry.npy": np.stack([s["geometry"] for s in samples]).astype(np.float32),
        "cot.npy": np.stack([s["cot"] for s in samples]).astype(np.float32),
        "ghi_wm2.npy": np.asarray([s["ghi"] for s in samples], dtype=np.float32),
        "clear_sky_wm2.npy": np.asarray([s["clear"] for s in samples], dtype=np.float32),
    }
    for name, arr in arrays.items():
        np.save(a.output / name, arr, allow_pickle=False)
    with (a.output / "rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("index", "split", "station", "time_bjt", "ghi_wm2", "clear_sky_wm2"))
        w.writeheader()
        for i, s in enumerate(samples):
            w.writerow({"index": i, "split": s["split"], "station": s["station"],
                        "time_bjt": s["time_bjt"], "ghi_wm2": s["ghi"],
                        "clear_sky_wm2": s["clear"]})
    counts = {sp: {st: sum(s["split"] == sp and s["station"] == st for s in samples)
                   for st in STATIONS} for sp in ("train", "val")}
    report = {
        "state": "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED",
        "samples": n, "samples_by_split_station": counts,
        "same_keys_all_arms": True,
        "label": "15-min right-closed GHI window; raw field retains the original engineering W/m2 operating convention; station sensor/export calibration metadata remains unverified",
        "target_time_basis": "BJT interval end", "split_sha256": SPLIT_SHA,
        "split_manifest_sha256": sha(a.split),
        "label_source_sha256": sha(a.labels),
        "feature_rows": len(feature_rows), "test_and_buffer_values_read": 0,
        "feature_report_sha256": input_hashes["feature_report"],
        "feature_array_sha256": input_hashes["features"],
        "feature_rows_sha256": input_hashes["feature_rows"],
        "r_bank_report_sha256": input_hashes["r_bank_report"],
        "r_prediction_sha256": input_hashes["r_prediction"],
        "r_rows_sha256": input_hashes["r_rows"],
        "r_checkpoint_sha256": r_report.get("frozen_R_checkpoint_sha256"),
        "r_model_source_sha256": r_report.get("model_source_sha256"),
        "r_norm_sha256": r_report.get("norm_sha256"),
        "source_precedence": "combined frozen train/validation AGRI and R-feature bank",
        "normalization": "not applied here; trainer fits per-station statistics on train rows only",
        "array_shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "array_sha256": {name: sha(a.output / name) for name in arrays},
        "rows_sha256": sha(a.output / "rows.csv"),
    }
    (a.output / "pack_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
