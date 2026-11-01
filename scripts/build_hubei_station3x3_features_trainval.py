"""Build station-centered 3x3 AGRI/geometry/frozen-COT input bank (train/val only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
STATIONS = {
    "Niushou": (8, 6),
    "Wangzhai": (9, 8),
    "Mizhuang": (7, 10),
}
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--prediction", required=True, type=Path)
    p.add_argument("--bank-report", required=True, type=Path)
    p.add_argument("--split-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.split_sha256 != EXPECTED_SPLIT_SHA:
        raise ValueError("Wrong frozen split SHA")
    report = json.loads(a.bank_report.read_text(encoding="utf-8"))
    if report.get("state") != "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED":
        raise ValueError("Full train/val frozen-R bank required")
    if report.get("test_payloads_read") != 0 or report.get("test_path_rows_in_manifest") != 0:
        raise ValueError("Test boundary violation")
    if report.get("prediction_sha256") != sha(a.prediction) or report.get("rows_sha256") != sha(a.rows):
        raise ValueError("R bank report does not bind the supplied prediction and row files")
    if report.get("manifest_sha256") != sha(a.manifest):
        raise ValueError("R bank report does not bind the supplied train/validation manifest")
    with a.rows.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    pred = np.load(a.prediction, mmap_mode="r", allow_pickle=False)
    if pred.shape != (len(rows), 1, 16, 16):
        raise ValueError(f"Unexpected bank shape {pred.shape}")
    if any(row["split"] not in ("train", "val") for row in rows):
        raise ValueError("Only frozen train/validation rows may be included")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    channels = np.lib.format.open_memmap(a.output / "station_features.npy", mode="w+", dtype=np.float32,
                                          shape=(len(rows), 3, 17, 3, 3))
    nonfinite_counts = np.zeros(17, dtype=np.int64)
    for i, row in enumerate(rows):
        with np.load(row["path_agri"], allow_pickle=True) as z:
            required = {"patch", "cosSOZ", "cosRAA", "day_mask"}
            if not required.issubset(z.files):
                raise ValueError(f"Missing feature key at {row['time_bjt']}")
            patch = np.asarray(z["patch"], dtype=np.float32)
            geo = [np.asarray(z[k], dtype=np.float32) for k in ("cosSOZ", "cosRAA", "day_mask")]
            if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
                raise ValueError(f"Bad AGRI shape at {row['time_bjt']}: {patch.shape}")
            if any(g.shape != (16, 16) for g in geo):
                raise ValueError(f"Bad geometry shape at {row['time_bjt']}")
            base = np.concatenate((patch[AGRI_CH], np.stack(geo)), axis=0)
        cot = np.asarray(pred[i, 0], dtype=np.float32)
        if not np.isfinite(cot).all() or (cot < 0).any():
            raise ValueError(f"Invalid COT bank row {i}")
        for si, (station, (r, c)) in enumerate(STATIONS.items()):
            if not (1 <= r < 15 and 1 <= c < 15):
                raise ValueError(f"3x3 crop out of bounds for {station}")
            sl = np.s_[r - 1:r + 2, c - 1:c + 2]
            channels[i, si, :16] = base[:, sl[0], sl[1]]
            channels[i, si, 16] = cot[sl]
        nonfinite_counts += (~np.isfinite(channels[i])).sum(axis=(0, 2, 3))
        if (i + 1) % 4096 == 0 or i + 1 == len(rows):
            print(f"PACKED {i+1}/{len(rows)}", flush=True)
    channels.flush()
    with (a.output / "rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "time_bjt", "split", "path_agri"])
        w.writeheader()
        w.writerows({"index": i, **r} for i, r in enumerate(rows))
    result = {
        "state": "HUBEI_STATION_3X3_RAW_FEATURES_TRAINVAL_COMPLETE_TEST_CLOSED",
        "rows_by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
        "stations": {k: {"center_local_16x16": list(v), "crop": "3x3 centered"} for k, v in STATIONS.items()},
        "input_channel_order": ["C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask", "R_COT"],
        "feature_shape": list(channels.shape), "feature_dtype": "float32",
        "nonfinite_feature_counts_by_channel": nonfinite_counts.tolist(),
        "R_COT_source": "frozen seed42 R; CPP fixed_v2 validation-selected; train rows are in-domain, not OOF",
        "test_payloads_read": 0, "split_sha256": a.split_sha256,
        "r_bank_report_sha256": sha(a.bank_report),
        "r_bank_prediction_sha256": sha(a.prediction),
        "r_bank_rows_sha256": sha(a.rows),
        "r_checkpoint_sha256": report.get("frozen_R_checkpoint_sha256"),
        "r_model_source_sha256": report.get("model_source_sha256"),
        "r_norm_sha256": report.get("norm_sha256"),
        "r_manifest_sha256": report.get("manifest_sha256"),
        "manifest_sha256": sha(a.manifest),
        "rows_sha256": sha(a.output / "rows.csv"), "features_sha256": sha(a.output / "station_features.npy"),
        "feature_file": "raw values, no GHI labels, no normalization fitted",
        "normalization_and_GHI_head_training": "not performed; unit/time and pvlib gates remain open",
    }
    (a.output / "FEATURE_REPORT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
