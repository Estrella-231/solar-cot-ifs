"""Independent NumPy/FP64 recomputation for a closed Hubei R validation run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run = Path(a.run)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "validation_fp64.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or status.get("test_payloads_read") != 0:
        raise ValueError("Run status is not closed validation-only completion")
    pred = np.load(run / "val_prediction_cot.npy", allow_pickle=False).astype(np.float64)
    truth = np.load(run / "val_reference_cpp_cot.npy", allow_pickle=False).astype(np.float64)
    mask = np.load(run / "val_valid_mask.npy", allow_pickle=False)
    if pred.shape != truth.shape or pred.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("Saved validation array shapes or mask type differ")
    with (run / "val_keys.csv").open(encoding="utf-8-sig", newline="") as f:
        keys = list(csv.DictReader(f))
    if len(keys) != len(pred) or any(row["split"] != "val" for row in keys):
        raise ValueError("Validation key manifest is inconsistent or contains non-val rows")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not mask.any():
        raise ValueError("Nonfinite saved validation values or empty mask")
    err = pred[mask] - truth[mask]
    y = truth[mask]
    ss_tot = np.sum((y - y.mean(dtype=np.float64)) ** 2, dtype=np.float64)
    calculated = {
        "valid_cpp_pixels": int(mask.sum()),
        "rmse_cot": float(np.sqrt(np.mean(err ** 2, dtype=np.float64))),
        "mae_cot": float(np.mean(np.abs(err), dtype=np.float64)),
        "bias_cot": float(np.mean(err, dtype=np.float64)),
        "r2_cot": float(1.0 - np.sum(err ** 2, dtype=np.float64) / ss_tot) if ss_tot > 0 else None,
    }
    expected = metrics["metrics"]
    for name, value in calculated.items():
        if isinstance(value, float):
            if abs(value - float(expected[name])) > 1e-7:
                raise ValueError(f"FP64 recomputation mismatch for {name}: {value} vs {expected[name]}")
        elif value != expected[name]:
            raise ValueError(f"Count mismatch for {name}: {value} vs {expected[name]}")
    report = {
        "state": "HUBEI_R_VALIDATION_FP64_INDEPENDENT_RECOMPUTE_PASSED_TEST_CLOSED",
        "test_payloads_read": 0,
        "split": "val only",
        "reference": "cpp_16_fixed_v2 retrieval, not independent truth",
        "val_rows_with_cpp_supervision": len(keys),
        "metrics": calculated,
        "matches_training_report_tolerance": 1e-7,
        "sha256": {name: sha(run / name) for name in (
            "best.pt", "val_prediction_cot.npy", "val_reference_cpp_cot.npy", "val_valid_mask.npy", "val_keys.csv")},
    }
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
