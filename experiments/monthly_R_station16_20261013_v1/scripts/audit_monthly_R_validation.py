#!/usr/bin/env python3
"""Independently rescore saved monthly-R validation predictions in FP64."""
import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scores(pred_log1p, ref, mask):
    m = np.asarray(mask, bool)
    if not m.any():
        raise ValueError("empty validation mask")
    p = np.expm1(np.asarray(pred_log1p, np.float64))
    y = np.asarray(ref, np.float64)
    e = p[m] - y[m]
    d = np.asarray(pred_log1p, np.float64)[m] - np.log1p(y[m])
    ad = np.abs(d)
    huber = np.where(ad <= .1, .5 * d * d, .1 * (ad - .05))
    return dict(valid_pixels=int(m.sum()), cot_rmse=float(np.sqrt(np.mean(e * e))),
                cot_mae=float(np.mean(np.abs(e))), cot_bias=float(np.mean(e)),
                huber_log1p=float(np.mean(huber)),
                thick_ge30_pixels=int(((y >= 30) & m).sum()),
                nearzero_lt0p1_pixels=int(((y < .1) & m).sum()),
                exact_zero_pixels=int(((y == 0) & m).sum()))


def stratified_scores(pred_log1p, ref, mask):
    p = np.expm1(np.asarray(pred_log1p, np.float64))
    y = np.asarray(ref, np.float64)
    m = np.asarray(mask, bool)
    bins = {
        "nearzero_[0,0.1)": m & (y < 0.1),
        "thin_[0.1,5)": m & (y >= 0.1) & (y < 5),
        "moderate_[5,30)": m & (y >= 5) & (y < 30),
        "thick_[30,100]": m & (y >= 30) & (y <= 100),
    }
    result = {}
    for name, bm in bins.items():
        if not bm.any():
            result[name] = dict(valid_pixels=0, cot_rmse=None, cot_mae=None,
                                cot_bias=None, mean_reference_cot=None,
                                mean_prediction_cot=None)
            continue
        error = p[bm] - y[bm]
        result[name] = dict(valid_pixels=int(bm.sum()),
                            cot_rmse=float(np.sqrt(np.mean(error * error))),
                            cot_mae=float(np.mean(np.abs(error))),
                            cot_bias=float(np.mean(error)),
                            mean_reference_cot=float(np.mean(y[bm])),
                            mean_prediction_cot=float(np.mean(p[bm])))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.run / "validation_predictions.npz", allow_pickle=False) as z:
        ids = z["sample_indices"].astype(np.int64)
        pred, ref, mask = z["pred_log1p_cot"], z["reference_cot"], z["mask"].astype(bool)
    if pred.shape != ref.shape or pred.shape != mask.shape or pred.ndim != 4 or pred.shape[1:] != (1, 16, 16):
        raise ValueError(f"validation array shape mismatch: {pred.shape} {ref.shape} {mask.shape}")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(pred[mask]).all() or not np.isfinite(ref[mask]).all():
        raise ValueError("duplicate validation ids or nonfinite saved values under mask")
    if np.any(pred[mask] < 0) or np.any(ref[mask] < 0) or np.any(ref[mask] > 100):
        raise ValueError("out-of-contract COT value under validation mask")
    with (a.pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    supervised = [r for r in rows if r["state"] == "SUPERVISED"]
    val_ids = np.asarray([i for i, r in enumerate(supervised) if r["split"] == "val"], np.int64)
    if not np.array_equal(ids, val_ids):
        raise ValueError("saved validation sample order differs from the pack row manifest")
    packed_cot = np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)[val_ids]
    packed_mask = np.load(a.pack / "mask.npy", mmap_mode="r", allow_pickle=False)[val_ids]
    if not np.array_equal(mask, packed_mask) or not np.array_equal(ref, packed_cot, equal_nan=True):
        raise ValueError("saved validation references/masks differ from the immutable packed arrays")
    all_scores = scores(pred, ref, mask)
    by_reference_bin = stratified_scores(pred, ref, mask)
    by_station = {}
    for station in ("sili", "zhujia"):
        sel = np.asarray([supervised[i]["station"] == station for i in ids])
        by_station[station] = dict(
            overall=scores(pred[sel], ref[sel], mask[sel]),
            by_reference_bin=stratified_scores(pred[sel], ref[sel], mask[sel]))
    stored = json.loads((a.run / "validation_metrics.json").read_text(encoding="utf-8"))
    metric_diffs = {k: abs(all_scores[k] - float(stored[k])) for k in ("cot_rmse", "cot_mae", "valid_pixels")}
    if metric_diffs["valid_pixels"] != 0 or max(metric_diffs["cot_rmse"], metric_diffs["cot_mae"]) > 1e-8:
        raise RuntimeError(f"stored validation metrics do not match FP64 rescoring: {metric_diffs}")
    report = dict(state="PASS_MONTHLY_R_VALIDATION_FP64_RESCORE", split_id="hunan_monthly_block_2d_buffer1d_seed42_v1",
                  all_pixels=all_scores, by_reference_bin=by_reference_bin, by_station=by_station,
                  compared_to_training_metrics_abs_diff=metric_diffs,
                  validation_prediction_sha256=sha(a.run / "validation_predictions.npz"),
                  checkpoint_sha256=sha(a.run / "best.pt"), train_pack_status_sha256=sha(a.pack / "pack_status.json"),
                  test_payloads_read=0, claim_scope="CPP-supported daytime retrieval-reference pixels; not independent truth")
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
