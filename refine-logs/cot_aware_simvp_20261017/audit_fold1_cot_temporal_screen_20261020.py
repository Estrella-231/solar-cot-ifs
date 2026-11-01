#!/usr/bin/env python3
"""Descriptive same-key COT trajectory screen for completed fold-1 OOF bank.

This reads only fold-1 OOF predicted COT plus the same sequences' two observed
history COT frames from the frozen-R base bank. It does not read future AGRI,
CPP, GHI, validation/test, or buffer payloads and is not a skill/gain estimate.
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
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_index(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary(v: np.ndarray) -> dict:
    x = np.asarray(v, dtype=np.float64).ravel()
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max_abs": float(np.max(np.abs(x))),
        "fraction_abs_gt_20": float(np.mean(np.abs(x) > 20)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", type=Path, required=True)
    p.add_argument("--history-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    fm = json.loads((a.forecast_bank / "bank_manifest.json").read_text())
    hm = json.loads((a.history_bank / "bank_manifest.json").read_text())
    assert fm["state"] == "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS" and fm["fold"] == 1
    assert fm["future_observed_agri_read"] is False and fm["cpp_read"] is False
    assert fm["validation_test_payload_opened"] is False and fm["buffer_payload_opened"] is False
    assert hm["state"] == "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK"
    assert hm["future_observed_agri_read"] is False and hm["cpp_read"] is False
    assert hm["buffer_payload_opened"] is False
    assert fm["r_checkpoint_sha256"] == hm["r_checkpoint_sha256"]

    fi, hi = read_index(a.forecast_bank / "index.csv"), read_index(a.history_bank / "index.csv")
    fmap = {r["seq_id"]: i for i, r in enumerate(fi)}
    # Store explicit lookup values because bank_index is the authoritative row.
    hrows = {r["seq_id"]: int(r["bank_index"]) for r in hi if r["split"] == "train"}
    ids = [r["seq_id"] for r in fi]
    if len(fmap) != len(fi) or len(hrows) != sum(r["split"] == "train" for r in hi):
        raise ValueError("duplicate sequence IDs in a bank")
    missing = [s for s in ids if s not in hrows]
    if missing:
        raise ValueError(f"history bank missing {len(missing)} fold-1 train keys; first={missing[:3]}")
    frows = np.asarray([int(r["bank_index"]) for r in fi], dtype=np.int64)
    hrows_i = np.asarray([hrows[s] for s in ids], dtype=np.int64)
    hist = np.load(a.history_bank / "history_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    pred = np.load(a.forecast_bank / "forecast_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if hist.shape[1:] != (2, 2, 1, 3, 3) or pred.shape[1:] != (2, 16, 1, 3, 3):
        raise ValueError(f"unexpected arrays: hist={hist.shape}; pred={pred.shape}")
    h = np.expm1(np.asarray(hist[hrows_i, :, :, 0], dtype=np.float32))
    q = np.expm1(np.asarray(pred[frows, :, :, 0], dtype=np.float32))
    if not np.isfinite(h).all() or not np.isfinite(q).all() or (h < 0).any() or (q < 0).any():
        raise ValueError("nonfinite or negative physical COT")

    stations = {}
    for si, name in enumerate(("sili", "zhujia")):
        # Two observed history scans, then 16 predicted scans; each adjacent
        # pair represents 15 minutes in this forecasting contract.
        path = np.concatenate((h[:, si, 1:2], q[:, si]), axis=1)
        d_hist = h[:, si, 1] - h[:, si, 0]
        d_pred = np.diff(path, axis=1)
        d_first = q[:, si, 0] - h[:, si, 1]
        stations[name] = {
            "history_last_15min_signed_delta": summary(d_hist),
            "forecast_first_step_signed_delta": summary(d_first),
            "forecast_15min_adjacent_signed_deltas_all_16_leads": summary(d_pred),
            "forecast_cot_physical_all_leads": summary(q[:, si]),
            "mean_abs_change_ratio_forecast_to_history": float(np.mean(np.abs(d_pred)) / max(np.mean(np.abs(d_hist)), 1e-12)),
        }
    out = {
        "state": "FOLD1_OOF_COT_TEMPORAL_DESCRIPTIVE_SCREEN",
        "interpretation_limit": "single heldout training-date fold; no CPP/GHI skill or causal gain claim",
        "inputs": {
            "audit_script_sha256": sha(Path(__file__)),
            "forecast_bank": str(a.forecast_bank),
            "forecast_manifest_sha256": sha(a.forecast_bank / "bank_manifest.json"),
            "history_bank": str(a.history_bank),
            "history_manifest_sha256": sha(a.history_bank / "bank_manifest.json"),
            "forecast_rows": int(len(fi)),
            "same_key_train_intersection": int(len(ids) - len(missing)),
            "shared_frozen_r_checkpoint_sha256": fm["r_checkpoint_sha256"],
            "history_source": "last two observed history AGRI frames processed by the same frozen R; history only",
            "future_observed_agri_read": False,
            "cpp_read": False,
            "ghi_read": False,
            "test_or_buffer_payload_read": False,
        },
        "units": "physical COT recovered from stored log1p(COT)",
        "stations": stations,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
