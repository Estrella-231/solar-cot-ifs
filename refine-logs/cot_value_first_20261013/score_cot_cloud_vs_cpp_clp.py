#!/usr/bin/env python3
"""Score the fixed COT>=0.1 cloud decision against CPP CLP on paired pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metric(pred, truth, mask):
    p, y = pred.astype(bool), truth.astype(bool)
    tp = int(np.count_nonzero(mask & p & y))
    fp = int(np.count_nonzero(mask & p & ~y))
    fn = int(np.count_nonzero(mask & ~p & y))
    tn = int(np.count_nonzero(mask & ~p & ~y))
    ratio = lambda n, d: (float(n / d) if d else None)
    pod_cloud = ratio(tp, tp + fn)
    pod_clear = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    return {
        "N": int(mask.sum()), "reference_cloud": tp + fn, "reference_clear": tn + fp,
        "predicted_cloud": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "POD_cloud": pod_cloud, "POD_clear": pod_clear,
        "precision_cloud": precision,
        "FAR_cloud": ratio(fp, tp + fp),
        "CSI_cloud": ratio(tp, tp + fp + fn),
        "HR_accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "KSS": (pod_cloud + pod_clear - 1.0 if pod_cloud is not None and pod_clear is not None else None),
        "frequency_bias_cloud": ratio(tp + fp, tp + fn),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--clp", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.predictions, allow_pickle=False) as z:
        ids = z["sample_indices"].astype(np.int64)
        pred = np.expm1(z["pred_log1p_cot"].astype(np.float64))[:, 0] >= 0.1
        cot_mask = z["mask"].astype(bool)[:, 0]
    with np.load(a.clp, allow_pickle=False) as z:
        clp, common, clp_ids = z["clp"][:, 0], z["common_mask"][:, 0].astype(bool), z["sample_indices"].astype(np.int64)
        stations, times = z["station"].astype(str), z["target_time_utc"].astype(str)
    if not np.array_equal(ids, clp_ids) or pred.shape != clp.shape or pred.shape != common.shape:
        raise ValueError("prediction and CLP validation keys/shapes differ")
    if not np.array_equal(cot_mask, common):
        raise ValueError("common-support mask is not identical to the saved R validation mask")
    if not np.isin(clp[common], (0, 1, 2)).all():
        raise ValueError("invalid CLP class on common support")
    truth_cloud = np.isin(clp, (1, 2))
    out = {
        "state": "PASS_PAIRED_COT_TO_CLP_CLOUD_CLASSIFICATION",
        "prediction": "monthly R physical COT >= 0.1",
        "reference": "CPP CLP; class 0 clear, 1 water cloud, 2 ice cloud",
        "common_support_pixels": int(common.sum()),
        "samples": int(len(ids)),
        "clp_class_counts": {str(k): int(np.count_nonzero(common & (clp == k))) for k in (0, 1, 2)},
        "overall": metric(pred, truth_cloud, common),
        "by_station": {},
        "test_payloads_read": 0,
        "caveat": "CPP CLP is a co-produced retrieval classification, not an independent cloud truth or the official FY-4B CLM mask.",
    }
    for st in ("sili", "zhujia"):
        sel = stations == st
        out["by_station"][st] = {"samples": int(sel.sum()), **metric(pred[sel], truth_cloud[sel], common[sel])}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
