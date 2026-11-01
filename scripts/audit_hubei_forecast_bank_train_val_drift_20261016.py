#!/usr/bin/env python3
"""Descriptive train/validation drift check for the strict Hubei forecast banks.

This reads only the frozen train/validation forecast AGRI and forecast-COT
banks plus their train-fitted AGRI normalization. It does not read test or
buffer payloads and makes no skill claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", type=Path, required=True)
    p.add_argument("--forecast-cot-bank", type=Path, required=True)
    p.add_argument("--agri-norm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    s_report = json.loads((a.forecast_bank / "BANK_REPORT.json").read_text())
    c_report = json.loads((a.forecast_cot_bank / "BANK_REPORT.json").read_text())
    if s_report.get("test_payloads_read") != 0 or c_report.get("test_payloads_read") != 0:
        raise ValueError("Expected train/validation-only forecast banks")
    if c_report.get("ghi_labels_read") is not False:
        raise ValueError("COT inference must not read GHI labels")
    if s_report.get("checkpoint_sha256") != c_report.get("source_simvp_checkpoint_sha256"):
        raise ValueError("S checkpoint lineage mismatch")
    norm = np.load(a.agri_norm, allow_pickle=False)
    std = np.maximum(np.asarray(norm["agri_std"], dtype=np.float64), 1e-6)
    if std.shape != (13,):
        raise ValueError(f"Unexpected channel std shape: {std.shape}")

    result: dict[str, object] = {
        "state": "HUBEI_FORECAST_BANK_TRAIN_VAL_DRIFT_DESCRIPTIVE",
        "test_and_buffer_payloads_read": 0,
        "ghi_labels_read": False,
        "forecast_train_is_oof": False,
        "validation_checkpoint_selected_by_validation": True,
        "s_checkpoint_sha256": s_report["checkpoint_sha256"],
        "agri_norm_scope": "frozen train-only channel standard deviations",
        "metric_scope": "all 13 channels, all 16 leads, full staged 16x16 field",
        "splits": {},
        "comparison_warning": (
            "Descriptive input-domain diagnostic only. Train forecast inputs are in-sample "
            "for S and train COT is inferred by frozen R on train-domain imagery; validation "
            "is out-of-sample for S/R. This is not GHI skill or an independent test."
        ),
    }
    for split in ("train", "val"):
        sd, cd = a.forecast_bank / split, a.forecast_cot_bank / split
        pred = np.load(sd / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
        truth = np.load(sd / "true_agri_patch.npy", mmap_mode="r", allow_pickle=False)
        cot = np.load(cd / "forecast_cot.npy", mmap_mode="r", allow_pickle=False)
        n = len(pred)
        if pred.shape != (n, 16, 13, 16, 16) or truth.shape != pred.shape:
            raise ValueError(f"Unexpected AGRI bank shape: {split} {pred.shape}/{truth.shape}")
        if cot.shape != (n, 16, 1, 16, 16):
            raise ValueError(f"Unexpected forecast COT shape: {split} {cot.shape}")
        leads = []
        sum_sq = sum_abs = sum_err = count = 0.0
        for lead_i in range(16):
            err = (np.asarray(pred[:, lead_i], dtype=np.float64) -
                   np.asarray(truth[:, lead_i], dtype=np.float64)) / std[None, :, None, None]
            leads.append({
                "lead_minutes": 15 * (lead_i + 1),
                "normalized_rmse": float(np.sqrt(np.mean(err * err))),
                "normalized_mae": float(np.mean(np.abs(err))),
                "normalized_bias": float(np.mean(err)),
                "sequence_count": n,
            })
            sum_sq += float(np.sum(err * err))
            sum_abs += float(np.abs(err).sum())
            sum_err += float(err.sum())
            count += err.size
        c = np.asarray(cot, dtype=np.float64)
        result["splits"][split] = {
            "sequences": n,
            "full_field_normalized_rmse": float(np.sqrt(sum_sq / count)),
            "full_field_normalized_mae": float(sum_abs / count),
            "full_field_normalized_bias": float(sum_err / count),
            "by_lead": leads,
            "forecast_cot": {
                "mean": float(c.mean()),
                "std": float(c.std()),
                "p50": float(np.quantile(c, 0.50)),
                "p90": float(np.quantile(c, 0.90)),
                "p99": float(np.quantile(c, 0.99)),
                "fraction_exact_zero": float(np.mean(c == 0)),
                "fraction_nonfinite": float(np.mean(~np.isfinite(c))),
            },
        }
    tr = result["splits"]["train"]["full_field_normalized_rmse"]
    va = result["splits"]["val"]["full_field_normalized_rmse"]
    result["val_to_train_rmse_ratio"] = float(va / tr) if tr > 0 else None
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
