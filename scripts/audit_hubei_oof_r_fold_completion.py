#!/usr/bin/env python3
"""Audit a completed Hubei OOF R fold without opening test/buffer payloads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--expected-pack", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    run, pack = a.run_dir, a.expected_pack
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    smoke = json.loads((run / "smoke_overfit.json").read_text(encoding="utf-8"))
    val_report = json.loads((run / "validation_fp64.json").read_text(encoding="utf-8"))
    pack_status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
    pack_hashes = json.loads((pack / "SHA256.json").read_text(encoding="utf-8"))
    actual_pack_hashes = {name: sha(pack / name) for name in pack_hashes}
    if actual_pack_hashes != pack_hashes or meta.get("source_pack_hashes") != pack_hashes:
        raise SystemExit("source pack SHA mismatch")
    if status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or status.get("test_payloads_read") != 0:
        raise SystemExit(f"run is not a completed test-closed validation run: {status}")
    if not smoke.get("passed") or smoke.get("updates") != 400:
        raise SystemExit(f"fixed-sample overfit gate not passed: {smoke}")
    if meta.get("allow_variable_support") is not True or meta.get("test_used") is not False:
        raise SystemExit("fold-specific support/test-closed contract mismatch")
    if pack_status.get("test_cpp_payloads_read") != 0 or pack_status.get("test_agri_payloads_read") != 0:
        raise SystemExit("test payload read recorded in pack")

    pred = np.load(run / "val_prediction_cot.npy", allow_pickle=False)
    ref = np.load(run / "val_reference_cpp_cot.npy", allow_pickle=False)
    mask = np.load(run / "val_valid_mask.npy", allow_pickle=False)
    with (run / "val_keys.csv").open(encoding="utf-8-sig", newline="") as f:
        keys = list(csv.DictReader(f))
    if pred.shape != (len(keys), 1, 16, 16) or ref.shape != pred.shape or mask.shape != pred.shape:
        raise SystemExit(f"validation array shape mismatch: {pred.shape}, {ref.shape}, {mask.shape}")
    expected_val_rows = int(meta["val_supervision_rows"])
    if len(keys) != expected_val_rows or len({row["time_bjt"] for row in keys}) != len(keys):
        raise SystemExit(f"validation keys must contain {expected_val_rows} unique supervised timestamps")
    if mask.dtype != np.bool_ or not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise SystemExit("validation arrays contain invalid dtype/nonfinite values")
    if (pred < 0).any() or ((ref[mask] < 0) | (ref[mask] > 100)).any() or not mask.any():
        raise SystemExit("COT physical range or validity mask failure")

    err = pred[mask].astype(np.float64) - ref[mask].astype(np.float64)
    truth = ref[mask].astype(np.float64)
    sst = np.sum((truth - truth.mean()) ** 2, dtype=np.float64)
    recomputed = {
        "valid_pixels": int(mask.sum()),
        "rmse_cot": float(np.sqrt(np.mean(err ** 2, dtype=np.float64))),
        "mae_cot": float(np.mean(np.abs(err), dtype=np.float64)),
        "bias_cot": float(np.mean(err, dtype=np.float64)),
        "r2_cot": float(1.0 - np.sum(err ** 2, dtype=np.float64) / sst) if sst > 0 else None,
    }
    recorded = val_report.get("metrics", {})
    if recorded.get("valid_cpp_pixels") != recomputed["valid_pixels"]:
        raise SystemExit("validation pixel count mismatch")
    for name in ("rmse_cot", "mae_cot", "bias_cot", "r2_cot"):
        if recorded.get(name) is None or abs(float(recorded[name]) - float(recomputed[name])) > 1e-5:
            raise SystemExit(f"independent FP64 recomputation mismatch in {name}")
    out = {
        "state": "HUBEI_OOF_R_FOLD_COMPLETION_AUDIT_PASSED_TEST_CLOSED",
        "run_dir": str(run),
        "pack_dir": str(pack),
        "pack_sha256": actual_pack_hashes,
        "best_checkpoint_sha256": sha(run / "best.pt"),
        "last_checkpoint_sha256": sha(run / "last.pt"),
        "validation_prediction_sha256": sha(run / "val_prediction_cot.npy"),
        "validation_reference_sha256": sha(run / "val_reference_cpp_cot.npy"),
        "validation_mask_sha256": sha(run / "val_valid_mask.npy"),
        "fold_seed": meta.get("seed"),
        "best_epoch": status.get("best_epoch"),
        "epochs_completed": status.get("epochs_completed"),
        "fixed_sample_gate": smoke,
        "validation_cpp_reference_not_independent_truth": recomputed,
        "test_payloads_read": 0,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
