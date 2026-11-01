#!/usr/bin/env python3
"""Extract one audited validation sequence and matching observed-future-R maps.

Reads only the corrected-solar full-train validation prediction bank and the
full-R validation prediction bank. It never opens train, test, or buffer rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-index", type=int, required=True)
    a = p.parse_args()
    sdir = a.base / "S_banks_attempt12" / "full" / "val"
    rdir = a.base / "R_runs_attempt12" / "full"
    cdir = a.base / "COT_banks_attempt12"
    skeys, rkeys = rows(sdir / "keys.csv"), rows(rdir / "val_keys.csv")
    assert all(r["split"] == "val" for r in skeys)
    assert all(r["split"] == "val" for r in rkeys)
    matches = [r for r in skeys if int(r["sample_index"]) == a.sample_index]
    if len(matches) != 1:
        raise ValueError(f"sample index must resolve uniquely, got {len(matches)}")
    srow = matches[0]
    target_times = srow["target_times_bjt"].split("|")
    if len(target_times) != 16:
        raise ValueError("Expected 16 canonical future target-time keys")
    r_lut = {r["time_bjt"]: i for i, r in enumerate(rkeys)}
    if len(r_lut) != len(rkeys):
        raise ValueError("Duplicate R validation timestamps")
    r_ids = [r_lut.get(t, -1) for t in target_times]
    if min(r_ids) < 0:
        raise ValueError("At least one target timestamp lacks offline R validation reference")

    pred = np.load(sdir / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(sdir / "true_agri_patch.npy", mmap_mode="r", allow_pickle=False)
    forecast_cot = np.load(cdir / "validation.npy", mmap_mode="r", allow_pickle=False)
    r_pred = np.load(rdir / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    r_mask = np.load(rdir / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    n = len(skeys)
    if pred.shape != (n, 16, 13, 16, 16) or truth.shape != pred.shape:
        raise ValueError(f"Unexpected AGRI bank shape: {pred.shape}, {truth.shape}")
    if forecast_cot.shape != (n, 16, 1, 16, 16):
        raise ValueError(f"Unexpected forecast COT bank shape: {forecast_cot.shape}")
    if r_pred.shape != r_mask.shape or r_pred.shape[1:] != (1, 16, 16):
        raise ValueError(f"Unexpected R validation arrays: {r_pred.shape}, {r_mask.shape}")

    lead_indices = np.array([3, 7, 11, 15], dtype=np.int64)  # +60, +120, +180, +240 min
    # AGRI channels are C01-C06, C09-C15; index 1 is C02 (0.65-um visible reflectance).
    observed_c02 = np.asarray(truth[a.sample_index, lead_indices, 1], dtype=np.float32)
    predicted_c02 = np.asarray(pred[a.sample_index, lead_indices, 1], dtype=np.float32)
    forecast_cot_maps = np.asarray(forecast_cot[a.sample_index, lead_indices, 0], dtype=np.float32)
    observed_future_cot_maps = np.asarray(r_pred[np.asarray(r_ids)[lead_indices], 0], dtype=np.float32)
    observed_future_cot_masks = np.asarray(r_mask[np.asarray(r_ids)[lead_indices], 0], dtype=bool)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        lead_minutes=np.array([60, 120, 180, 240], dtype=np.int16),
        target_time_bjt=np.array([target_times[i] for i in lead_indices]),
        observed_c02=observed_c02,
        predicted_c02=predicted_c02,
        forecast_cot=forecast_cot_maps,
        observed_future_agri_to_r_cot=observed_future_cot_maps,
        observed_future_cot_valid_mask=observed_future_cot_masks,
    )
    manifest = {
        "state": "VALIDATION_ONLY_CASE_ARRAYS_EXTRACTED",
        "sample_index": a.sample_index,
        "init_time_bjt": srow["init_time_bjt"],
        "target_times_bjt": [target_times[i] for i in lead_indices],
        "lead_minutes": [60, 120, 180, 240],
        "split": "validation",
        "agri_visual_channel": "C02; preprocessed AGRI visible reflectance; bank values are denormalized SimVP outputs for predictions",
        "forecast_cot_source": str(cdir / "validation.npy"),
        "offline_reference": "corrected frozen R applied to observed future AGRI at target time; retrieval-chain reference, not independent COT truth and not deployable",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "source_sha256": {
            "S_keys": sha(sdir / "keys.csv"),
            "S_prediction_bank": sha(sdir / "pred_agri_patch.npy"),
            "S_observed_bank": sha(sdir / "true_agri_patch.npy"),
            "forecast_COT_bank": sha(cdir / "validation.npy"),
            "R_validation_keys": sha(rdir / "val_keys.csv"),
            "R_validation_prediction": sha(rdir / "val_prediction_cot.npy"),
            "R_validation_mask": sha(rdir / "val_valid_mask.npy"),
        },
    }
    a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
