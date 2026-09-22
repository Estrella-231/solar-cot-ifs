"""Read-only validation error decomposition for the frozen Hunan COT retriever."""
import json
from pathlib import Path

import numpy as np


ROOT = Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907")
EDGES = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0001)


def summarize(reference, predicted, valid):
    count = int(valid.sum())
    if count == 0:
        return {"pixels": 0}
    residual = (predicted - reference)[valid].astype(np.float64)
    return {"pixels": count, "reference_mean": float(reference[valid].mean()),
            "predicted_mean": float(predicted[valid].mean()),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "bias": float(residual.mean())}


def main():
    path = ROOT / "runs/cot_repaired_seed42_v1/validation_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        rows = saved["row_indices"]
        reference = saved["reference_cot"].astype(np.float32)[:, 0]
        predicted = np.expm1(saved["pred_log1p_cot"].astype(np.float64)[:, 0]).astype(np.float32)
        mask = saved["mask"][:, 0].astype(bool)
    assert len(rows) == 6536 and reference.shape == predicted.shape == mask.shape
    assert np.isfinite(reference).all() and np.isfinite(predicted).all()
    center = np.zeros((16, 16), dtype=bool)
    center[6:11, 6:11] = True
    result = {"state": "COMPLETE_VALIDATION_ONLY_COT_RETRIEVAL_ERROR_DECOMPOSITION",
              "test_used": False, "rows": len(rows), "all": summarize(reference, predicted, mask),
              "cot_bins": {}, "spatial": {
                  "station_5x5": summarize(reference, predicted, mask & center[None]),
                  "outside_5x5": summarize(reference, predicted, mask & ~center[None])},
              "thresholds": {}}
    for lower, upper in zip(EDGES[:-1], EDGES[1:]):
        key = f"[{lower:g},{upper:g})"
        result["cot_bins"][key] = summarize(reference, predicted,
                                                mask & (reference >= lower) & (reference < upper))
    for threshold in (1.0, 3.0, 10.0):
        truth_cloud = mask & (reference >= threshold)
        predicted_cloud = mask & (predicted >= threshold)
        hit = int((truth_cloud & predicted_cloud).sum())
        false_alarm = int((~truth_cloud & predicted_cloud).sum())
        result["thresholds"][str(threshold)] = {
            "reference_cloud_pixels": int(truth_cloud.sum()),
            "predicted_cloud_pixels": int(predicted_cloud.sum()),
            "recall": hit / max(int(truth_cloud.sum()), 1),
            "precision": hit / max(hit + false_alarm, 1)}
    out = ROOT / "audits/cot_retriever_bins_20260922.json"
    if out.exists():
        raise RuntimeError("refuse to overwrite prior COT diagnostic")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
