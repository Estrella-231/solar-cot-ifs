"""Build Hubei fixed-v2 COT-R train/validation arrays only; never load test paths."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
SPLIT_ORDER = {"train": 0, "val": 1}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-split-sha256", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    with open(a.manifest, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["split"] not in SPLIT_ORDER for r in rows):
        raise ValueError("Manifest must contain only non-empty frozen train/val rows")
    rows.sort(key=lambda r: (SPLIT_ORDER[r["split"]], r["time_bjt"]))
    times = [r["time_bjt"] for r in rows]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate train/validation time key")
    n = len(rows)
    x = np.empty((n, 16, 16, 16), dtype=np.float32)
    target = np.zeros((n, 1, 16, 16), dtype=np.float32)
    mask = np.zeros((n, 1, 16, 16), dtype=np.bool_)
    per_split = Counter()
    valid_pixels = Counter()
    nan_pixels = Counter()
    invalid_target_pixels = Counter()
    source_meta = []
    for i, row in enumerate(rows):
        with np.load(row["path_agri"], allow_pickle=True) as z:
            required = {"patch", "cosSOZ", "cosRAA", "day_mask"}
            if not required.issubset(z.files):
                raise ValueError(f"Missing AGRI geometry key at {row['time_bjt']}")
            patch = np.asarray(z["patch"], dtype=np.float32)
            if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
                raise ValueError(f"Unexpected AGRI patch shape {patch.shape} at {row['time_bjt']}")
            channels = np.concatenate((patch[AGRI_CH], np.asarray(z["cosSOZ"], np.float32)[None],
                                       np.asarray(z["cosRAA"], np.float32)[None],
                                       np.asarray(z["day_mask"], np.float32)[None]), axis=0)
            if channels.shape != (16, 16, 16):
                raise ValueError(f"Unexpected 16-channel input shape {channels.shape}")
            x[i] = channels
        cpp = np.load(row["path_cpp_fixed_v2"], allow_pickle=False)
        if cpp.ndim != 3 or cpp.shape[0] < 1 or cpp.shape[1:] != (16, 16):
            raise ValueError(f"Unexpected fixed_v2 CPP shape {cpp.shape} at {row['time_bjt']}")
        cot = np.asarray(cpp[0], dtype=np.float32)
        valid = np.isfinite(cot) & (cot >= 0.0) & (cot <= 100.0)
        target[i, 0] = np.where(valid, cot / 100.0, 0.0)
        mask[i, 0] = valid
        per_split[row["split"]] += 1
        valid_pixels[row["split"]] += int(valid.sum())
        nan_pixels[row["split"]] += int((~np.isfinite(cot)).sum())
        invalid_target_pixels[row["split"]] += int((~valid).sum())
        source_meta.append({"time_bjt": row["time_bjt"], "split": row["split"],
                            "agri_path": row["path_agri"], "cpp_path": row["path_cpp_fixed_v2"]})
        if (i + 1) % 500 == 0:
            print(f"PACKED {i+1}/{n}", flush=True)

    train = np.asarray([r["split"] == "train" for r in rows])
    if not train.any() or not np.asarray([r["split"] == "val" for r in rows]).any():
        raise ValueError("Both train and validation must be non-empty")
    mean = np.zeros(16, np.float64)
    std = np.ones(16, np.float64)
    counts = np.zeros(16, np.int64)
    for c in range(16):
        vals = x[train, c].astype(np.float64)
        finite = np.isfinite(vals)
        counts[c] = int(finite.sum())
        if not counts[c]:
            raise ValueError(f"No finite training values in input channel {c}")
        clean = vals[finite]
        mean[c] = clean.mean(dtype=np.float64)
        std[c] = max(float(clean.std(dtype=np.float64)), 1e-6)
    x_norm = (x - mean.astype(np.float32)[None, :, None, None]) / std.astype(np.float32)[None, :, None, None]
    nonfinite_inputs = int((~np.isfinite(x_norm)).sum())
    x_norm[~np.isfinite(x_norm)] = 0.0
    arrays = {"x_norm.npy": x_norm.astype(np.float32), "target.npy": target, "mask.npy": mask}
    for name, arr in arrays.items():
        np.save(out / name, arr, allow_pickle=False)
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "split", "time_bjt", "path_agri", "path_cpp_fixed_v2"])
        w.writerows((i, r["split"], r["time_bjt"], r["path_agri"], r["path_cpp_fixed_v2"]) for i, r in enumerate(rows))
    norm = {"input_channels": ["C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask"],
            "mean": mean.tolist(), "std": std.tolist(), "count_finite_train_pixels_by_channel": counts.tolist(),
            "fit_split": "train only", "train_samples": int(train.sum()), "test_used": False}
    (out / "norm.json").write_text(json.dumps(norm, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    report = {
        "state": "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED",
        "source_split_sha256": a.source_split_sha256,
        "path_manifest_sha256": sha(a.manifest),
        "n_samples": n,
        "counts_by_split": {k: int(per_split[k]) for k in ("train", "val")},
        "valid_cpp_pixels_by_split": {k: int(valid_pixels[k]) for k in ("train", "val")},
        "nonfinite_cpp_pixels_by_split": {k: int(nan_pixels[k]) for k in ("train", "val")},
        "invalid_cpp_target_pixels_by_split": {k: int(invalid_target_pixels[k]) for k in ("train", "val")},
        "test_cpp_payloads_read": 0,
        "test_agri_payloads_read": 0,
        "buffer_cpp_payloads_read": 0,
        "buffer_agri_payloads_read": 0,
        "normalization_fit": "train only",
        "nonfinite_normalized_input_values_imputed_to_zero": nonfinite_inputs,
        "model_contract": "16 channels: 13 AGRI + cosSOZ + cosRAA + day_mask; CPP plane 0 finite [0,100] mask; target=COT/100, train objective will use log1p(COT)",
        "target_invalid_pixels": "zero target with false mask; not interpreted as clear sky",
        "source_files": {"rows.csv": sha(out / "rows.csv"), **hashes},
        "samples_source_metadata": source_meta,
    }
    (out / "pack_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("state", "counts_by_split", "valid_cpp_pixels_by_split", "nonfinite_cpp_pixels_by_split", "test_cpp_payloads_read", "nonfinite_normalized_input_values_imputed_to_zero")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
