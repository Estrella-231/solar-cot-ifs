"""Independently verify station-crop keys and center COT against the frozen bank."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STATIONS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--bank-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    fr = json.loads((a.features_dir / "FEATURE_REPORT.json").read_text(encoding="utf-8"))
    br = json.loads((a.bank_dir / "BANK_REPORT.json").read_text(encoding="utf-8"))
    if fr.get("test_payloads_read") != 0 or br.get("test_payloads_read") != 0:
        raise ValueError("Test boundary violation")
    feature_rows, bank_rows = read_rows(a.features_dir / "rows.csv"), read_rows(a.bank_dir / "rows.csv")
    if [(r["time_bjt"], r["split"]) for r in feature_rows] != [(r["time_bjt"], r["split"]) for r in bank_rows]:
        raise ValueError("Feature rows do not match bank keys")
    x = np.load(a.features_dir / "station_features.npy", mmap_mode="r", allow_pickle=False)
    cot = np.load(a.bank_dir / "cot_pred_trainval_full.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (len(feature_rows), 3, 17, 3, 3) or cot.shape != (len(bank_rows), 1, 16, 16):
        raise ValueError(f"Shape mismatch: features={x.shape}, COT={cot.shape}")
    max_center_err = 0.0
    n = 0
    for i in range(len(feature_rows)):
        for si, (r, c) in enumerate(STATIONS.values()):
            err = abs(float(x[i, si, 16, 1, 1]) - float(cot[i, 0, r, c]))
            max_center_err = max(max_center_err, err)
            n += 1
    if max_center_err != 0.0:
        raise ValueError(f"Station center COT differs from full bank: {max_center_err}")
    # COT sentinels must coincide with the stored local solar mask being off.
    off = np.asarray(x[:, :, 15] <= 0.5)
    cot_crop = np.asarray(x[:, :, 16])
    if np.any(cot_crop[off] != 0):
        raise ValueError("Nonzero COT sentinel where station day_mask is off")
    result = {
        "state": "PASS_STATION_CROP_KEYS_SHAPES_CENTER_COT_TEST_CLOSED",
        "rows_by_split": {s: sum(r["split"] == s for r in feature_rows) for s in ("train", "val")},
        "feature_shape": list(x.shape), "feature_dtype": str(x.dtype),
        "all_features_finite": bool(np.isfinite(x).all()),
        "center_cot_exact_match_to_full_bank": max_center_err == 0.0,
        "center_samples_checked": n, "max_center_abs_difference": max_center_err,
        "daymask_off_cot_zero_sentinel_pass": True,
        "station_centers_local_16x16": {k: list(v) for k, v in STATIONS.items()},
        "features_sha256": sha(a.features_dir / "station_features.npy"),
        "rows_sha256": sha(a.features_dir / "rows.csv"),
        "test_payloads_read": 0, "GHI_values_used": False,
        "COT_train_predictions_are_OOF": False,
        "GHI_unit_time_pvlib_gates": "still open; no training authorized by this QA",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
