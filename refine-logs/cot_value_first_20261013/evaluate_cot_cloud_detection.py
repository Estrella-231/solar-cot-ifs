#!/usr/bin/env python3
"""Fixed-threshold cloud/thick-COT detection audit for monthly R validation.

Uses the saved, independently FP64-audited validation predictions and CPP
reference COT. It never reads test payloads or treats missing pixels as clear.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def scores(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    y = ref >= threshold
    p = pred >= threshold
    tp = int(np.count_nonzero(mask & y & p))
    fp = int(np.count_nonzero(mask & ~y & p))
    fn = int(np.count_nonzero(mask & y & ~p))
    tn = int(np.count_nonzero(mask & ~y & ~p))
    div = lambda a, b: float(a / b) if b else None
    return {
        "threshold_cot": threshold,
        "valid_pixels": int(mask.sum()),
        "reference_positive": tp + fn,
        "predicted_positive": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pod_recall": div(tp, tp + fn),
        "precision": div(tp, tp + fp),
        "false_alarm_ratio": div(fp, tp + fp),
        "csi": div(tp, tp + fp + fn),
        "frequency_bias": div(tp + fp, tp + fn),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    with np.load(a.run / "validation_predictions.npz", allow_pickle=False) as z:
        ids = z["sample_indices"].astype(np.int64)
        pred = np.expm1(z["pred_log1p_cot"].astype(np.float64))
        ref = z["reference_cot"].astype(np.float64)
        mask = z["mask"].astype(bool)
    if pred.shape != ref.shape or pred.shape != mask.shape or pred.ndim != 4:
        raise ValueError("prediction, reference, and mask shapes do not agree")
    if not np.isfinite(pred[mask]).all() or not np.isfinite(ref[mask]).all():
        raise ValueError("nonfinite values under valid CPP mask")

    with (a.pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["state"] == "SUPERVISED"]
    val_ids = np.asarray([i for i, r in enumerate(rows) if r["split"] == "val"], dtype=np.int64)
    if not np.array_equal(ids, val_ids):
        raise ValueError("prediction sample order does not match immutable validation rows")
    if not np.array_equal(mask, np.load(a.pack / "mask.npy", mmap_mode="r", allow_pickle=False)[ids]):
        raise ValueError("saved validation mask differs from reference pack")
    if not np.array_equal(ref, np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)[ids], equal_nan=True):
        raise ValueError("saved validation CPP reference differs from immutable pack")

    valid = mask
    out = {
        "state": "PASS_FIXED_THRESHOLD_COT_DETECTION",
        "split": "validation_only",
        "sample_count": int(len(ids)),
        "valid_pixels": int(valid.sum()),
        "reference": "CPP COT labels; retrieval-reference, not independent truth",
        "cloud_definition": "COT >= 0.1",
        "thick_cloud_definition": "COT >= 30",
        "prediction_transform": "expm1(log1p_COT), applied once",
        "thresholds_selected_on_validation": False,
        "missing_or_invalid_cpp_as_clear": False,
        "test_payloads_read": 0,
        "overall": {
            "cloud": scores(pred, ref, valid, 0.1),
            "thick_cloud": scores(pred, ref, valid, 30.0),
        },
        "by_station": {},
    }
    station_ids = np.asarray([r["station"].lower() for r in rows])
    for station in ("sili", "zhujia"):
        sel = station_ids[ids] == station
        out["by_station"][station] = {
            "samples": int(sel.sum()),
            "cloud": scores(pred[sel], ref[sel], valid[sel], 0.1),
            "thick_cloud": scores(pred[sel], ref[sel], valid[sel], 30.0),
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
