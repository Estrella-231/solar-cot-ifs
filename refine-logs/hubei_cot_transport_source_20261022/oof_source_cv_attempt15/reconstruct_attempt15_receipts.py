#!/usr/bin/env python3
"""Reconstruct post-fit receipts for attempt15 seed42/43 after a metadata-only crash."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    attempt = Path(sys.argv[1]).resolve()
    cache_build = Path(sys.argv[2]).resolve()
    runroot = attempt / "training"
    arm_cache = {
        "fullr_control": cache_build / "fullr_control" / "cache",
        "oof_train_source": cache_build / "oof_train_source" / "cache",
    }
    run_keys: dict[tuple[str, int], list[str]] = {}

    for arm in arm_cache:
        cache_report_path = arm_cache[arm] / "cache_report.json"
        cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
        for seed in (42, 43):
            out = runroot / f"{arm}_seed{seed}"
            best_path, last_path = out / "best.pt", out / "last.pt"
            pred_path, log_path = out / "outer_train_holdout_predictions.npz", out / "training.jsonl"
            assert all(p.is_file() for p in (best_path, last_path, pred_path, log_path)), out
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [int(r["epoch"]) for r in records] == list(range(1, 11)), (out, len(records))
            steps = [int(r["global_step"]) for r in records]
            assert steps[-1] == 1350 and all(a < b for a, b in zip(steps, steps[1:])), (out, steps)
            assert all(int(r["global_batch"]) == 32 and int(r["world_size"]) == 1 for r in records)

            best = torch.load(best_path, map_location="cpu", weights_only=False)
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            assert int(best["seed"]) == seed and int(last["seed"]) == seed
            assert int(best["epoch"]) in [int(r["epoch"]) for r in records]
            assert int(last["epoch"]) == 10 and int(last["global_step"]) == 1350
            assert int(best["global_batch"]) == 32 and int(last["global_batch"]) == 32
            assert int(best["world_size"]) == 1 and int(last["world_size"]) == 1

            with np.load(pred_path, allow_pickle=False) as pred:
                keys = pred["init_time_bjt"].astype(str).tolist()
                values = pred["pred_log_scaled"]
                assert len(keys) == len(set(keys)) and values.shape == (len(keys), 16, 1, 16, 16), (out, values.shape)
                assert np.isfinite(values).all(), out
            run_keys[(arm, seed)] = keys

            receipt = {
                "state": "COMPLETE_TRAINED_HUBEI_INTERNAL_TRAIN_DATE_HOLDOUT_COT_DIAGNOSTIC",
                "receipt_mode": "reconstructed_after_postfit_reporting_error",
                "receipt_note": "All 10 epoch records, last checkpoint, best checkpoint and saved outer predictions were present. The original PBS task exited 1 only while writing this metadata because report was undefined in train(); no optimizer step was resumed or repeated.",
                "original_pbs_job_id": "215542.tc6000",
                "original_pbs_exit_status": 1,
                "scope": "Hubei shared 16x16 station-neighborhood patch; outer evaluation uses original train dates only, not full region or independent test.",
                "split": "internal train-date holdout from frozen train dates; official monthly validation, test, and buffer not read",
                "selected_checkpoint": "best.pt",
                "selected_epoch": int(best["epoch"]),
                "global_step": int(last["global_step"]),
                "world_size": 1,
                "global_batch": 32,
                "seed": seed,
                "fit_windows": int(records[-1]["fit_windows"]),
                "inner_holdout_windows": int(records[-1]["inner_holdout_windows"]),
                "train_config": {
                    "width": 96, "max_epochs": 10, "patience": 11, "min_delta": 1e-3,
                    "learning_rate": 1e-4, "minimum_learning_rate": 1e-6,
                    "weight_decay": 1e-4, "loader_workers": 0,
                },
                "inner_split": best["split_report"],
                "inner_holdout_support": best["inner_holdout_support"],
                "training_history_source": cache_report.get("history_source", cache_report.get("history_cot_source")),
                "outer_holdout_history_source": "final frozen full-R history in both arms",
                "cpp_role": "retrieval reference, not independent cloud truth",
                "test_payloads_read": 0,
                "buffer_payloads_read": 0,
                "official_monthly_validation_payloads_read": 0,
                "outer_train_holdout_init_windows": len(keys),
                "epoch_log_sha256": sha256(log_path),
                "best_checkpoint_sha256": sha256(best_path),
                "last_checkpoint_sha256": sha256(last_path),
                "outer_holdout_predictions_sha256": sha256(pred_path),
                "cache_report_sha256": sha256(cache_report_path),
            }
            receipt_path = out / "training_complete.json"
            if receipt_path.exists():
                raise FileExistsError(f"refusing to replace existing receipt: {receipt_path}")
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            files = ("best.pt", "last.pt", "outer_train_holdout_predictions.npz", "training.jsonl", "training_complete.json")
            (out / "SHA256SUMS").write_text(
                "\n".join(f"{sha256(out / name)}  {name}" for name in files) + "\n", encoding="utf-8")

    key_sets = [set(v) for v in run_keys.values()]
    assert key_sets and all(keys == key_sets[0] for keys in key_sets[1:]), "first-wave outer init keys differ"
    print(json.dumps({"state": "PASS_RECONSTRUCTED_4_FIRST_WAVE_RECEIPTS",
                      "runs": len(run_keys), "common_outer_init_windows": len(key_sets[0]),
                      "optimizer_steps_per_run": 1350, "global_batch": 32,
                      "test_payloads_read": 0, "buffer_payloads_read": 0}, indent=2))


if __name__ == "__main__":
    main()
