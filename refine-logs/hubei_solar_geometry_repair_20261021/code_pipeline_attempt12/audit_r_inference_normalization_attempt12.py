"""Read-only check that future-AGRI R inference uses the trained 16-channel transform."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = a.base / "prepared_attempt12"
    raw = np.load(data / "R_raw_corrected_canonical_attempt12.npy", mmap_mode="r", allow_pickle=False)
    packed = np.load(data / "R_packs/full/x_norm.npy", mmap_mode="r", allow_pickle=False)
    norm = json.loads((data / "R_packs/full/norm.json").read_text(encoding="utf-8"))
    with (data / "R_packs/full/rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if raw.shape != packed.shape or raw.shape != (len(rows), 16, 16, 16):
        raise ValueError(f"R shape mismatch raw={raw.shape}, packed={packed.shape}, rows={len(rows)}")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("Unexpected split in R train/validation pack")
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    max_abs = 0.0
    old_override_max_abs = 0.0
    n_day = n_night = 0
    for start in range(0, len(rows), 128):
        stop = min(start + 128, len(rows))
        x = np.asarray(raw[start:stop], dtype=np.float32)
        expected = (x - mean) / std
        got = np.asarray(packed[start:stop], dtype=np.float32)
        max_abs = max(max_abs, float(np.max(np.abs(expected.astype(np.float64) - got.astype(np.float64)))))
        day = x[:, 15:16] > 0.5
        n_day += int(day.sum())
        n_night += int((~day).sum())
        old_override = expected.copy()
        old_override[:, 15:16] = np.where(day, expected[:, 15:16], 0.0)
        old_override_max_abs = max(old_override_max_abs, float(np.max(np.abs(old_override.astype(np.float64) - got.astype(np.float64)))))
    report = {
        "state": "PASS_R_INFERENCE_NORMALIZATION_MATCHES_TRAIN_VALIDATION_PACK",
        "rows": len(rows),
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val")},
        "input_shape": list(raw.shape),
        "channel_order": norm["input_channels"],
        "train_fit_normalization_sha256_file": str(data / "R_packs/full/norm.json"),
        "new_exact_train_transform_max_abs": max_abs,
        "legacy_binary_daymask_override_max_abs_vs_train_transform": old_override_max_abs,
        "day_pixels": n_day,
        "night_pixels": n_night,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "interpretation": "Future-AGRI inference must apply the full train-fitted z-score to all 16 channels, including day_mask; only the R output is zeroed outside day_mask>0.5 and cosSOZ>cos(80deg).",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if max_abs > 1e-6:
        raise SystemExit("R input transform does not exactly match the prepared training pack")


if __name__ == "__main__":
    main()
