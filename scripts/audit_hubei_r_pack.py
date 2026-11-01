"""Verify saved Hubei R train/validation pack contents and hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    pack = Path(a.pack)
    status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
    expected = json.loads((pack / "SHA256.json").read_text(encoding="utf-8"))
    hash_actual = {n: sha(pack / n) for n in expected}
    if hash_actual != expected:
        raise ValueError("Pack SHA256 mismatch")
    with (pack / "rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.load(pack / "x_norm.npy", allow_pickle=False, mmap_mode="r")
    y = np.load(pack / "target.npy", allow_pickle=False, mmap_mode="r")
    m = np.load(pack / "mask.npy", allow_pickle=False, mmap_mode="r")
    n = len(rows)
    if x.shape != (n, 16, 16, 16) or y.shape != (n, 1, 16, 16) or m.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{x.shape}, y{y.shape}, m{m.shape}, rows={n}")
    splits = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "val")}
    if sum(map(len, splits.values())) != n or n != status["n_samples"]:
        raise ValueError("Split manifest count mismatch")
    if m.dtype != np.bool_ or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid tensor dtype or nonfinite values")
    if (y < 0).any() or (y > 1).any() or (y[~m] != 0).any():
        raise ValueError("Target/mask contract violation")
    valid_pixels = {s: int(m[ids].sum()) for s, ids in splits.items()}
    valid_per_sample = m.reshape(n, -1).sum(axis=1)
    samples_with_valid_mask = {
        s: int((valid_per_sample[ids] > 0).sum()) for s, ids in splits.items()
    }
    samples_without_valid_mask = {
        s: int((valid_per_sample[ids] == 0).sum()) for s, ids in splits.items()
    }
    min_valid_pixels_per_sample = {
        s: int(valid_per_sample[ids].min()) for s, ids in splits.items() if ids
    }
    if valid_pixels != status["valid_cpp_pixels_by_split"]:
        raise ValueError("Mask counts differ from build report")
    norm = json.loads((pack / "norm.json").read_text(encoding="utf-8"))
    if norm["train_samples"] != len(splits["train"]) or norm["test_used"] is not False:
        raise ValueError("Normalization split provenance mismatch")
    report = {"status": "HUBEI_R_PACK_CPU_INTEGRITY_AUDIT_PASSED_TEST_CLOSED",
              "test_payloads_read": 0,
              "pack_status_sha256": sha(pack / "pack_status.json"),
              "hashes_match": True,
              "counts_by_split": {s: len(ids) for s, ids in splits.items()},
              "input_shape": list(x.shape), "target_shape": list(y.shape), "mask_shape": list(m.shape),
              "mask_dtype": str(m.dtype), "input_finite": True, "target_finite": True,
              "valid_cpp_pixels_by_split": valid_pixels,
              "samples_with_valid_cpp_pixels_by_split": samples_with_valid_mask,
              "samples_without_valid_cpp_pixels_by_split": samples_without_valid_mask,
              "min_valid_cpp_pixels_per_sample_by_split": min_valid_pixels_per_sample,
              "train_only_norm_samples": norm["train_samples"],
              "masked_target_nonzero_count": int((y[~m] != 0).sum()),
              "all_samples_have_valid_cpp_pixel": bool(m.reshape(n, -1).any(axis=1).all())}
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
