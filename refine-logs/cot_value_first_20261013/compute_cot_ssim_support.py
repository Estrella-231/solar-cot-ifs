#!/usr/bin/env python3
"""Compute train-only COT SSIM scale and CPP-supported validation SSIM."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gaussian_kernel(size=11, sigma=1.5):
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    g /= g.sum()
    return np.outer(g, g)


def ssim_valid(pred, truth, mask, kernel, data_range=1.0):
    """SSIM for all fully valid 11x11 windows, using population moments."""
    from numpy.lib.stride_tricks import sliding_window_view

    p = sliding_window_view(pred, kernel.shape)
    t = sliding_window_view(truth, kernel.shape)
    m = sliding_window_view(mask, kernel.shape)
    good = m.all(axis=(-2, -1))
    if not good.any():
        return np.empty(0, dtype=np.float64)
    mu_p = np.einsum("ij,xyij->xy", kernel, p, optimize=True)
    mu_t = np.einsum("ij,xyij->xy", kernel, t, optimize=True)
    var_p = np.einsum("ij,xyij->xy", kernel, p * p, optimize=True) - mu_p * mu_p
    var_t = np.einsum("ij,xyij->xy", kernel, t * t, optimize=True) - mu_t * mu_t
    cov = np.einsum("ij,xyij->xy", kernel, p * t, optimize=True) - mu_p * mu_t
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (var_p + var_t + c2)
    )
    return np.asarray(score[good], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = list(csv.DictReader((a.pack / "rows.csv").open(encoding="utf-8", newline="")))
    train_ids = np.asarray([
        int(r["array_index"]) for r in rows
        if r["split"] == "train" and r["state"] == "SUPERVISED"
    ], dtype=np.int64)
    if len(train_ids) == 0:
        raise RuntimeError("no supervised train CPP references")
    cot_path, mask_path = a.pack / "cot.npy", a.pack / "mask.npy"
    cot = np.load(cot_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False).astype(bool)
    expected_examples = sum(r["state"] == "SUPERVISED" for r in rows)
    if cot.shape != (expected_examples, 1, 16, 16) or mask.shape != cot.shape:
        raise RuntimeError(f"unexpected train CPP pack shapes: {cot.shape}/{mask.shape}")
    y = np.asarray(cot[train_ids, 0], dtype=np.float64)
    m = mask[train_ids, 0]
    vals = y[m]
    if not np.isfinite(vals).all() or np.any(vals < 0) or np.any(vals > 100):
        raise RuntimeError("train CPP references violate physical range/finite contract")
    q999 = float(np.quantile(vals, 0.999, method="linear"))
    if not q999 > 0:
        raise RuntimeError("train-only q99.9 scale is non-positive")
    L = float(np.log1p(q999))

    with np.load(a.bank, allow_pickle=False) as z:
        pred_log = z["forecast_log1p_cot"].astype(np.float64)
        persist_log = z["persistence_log1p_cot"].astype(np.float64)
        ref = z["cpp_reference_cot"].astype(np.float64)
        vmask = z["cpp_valid_mask"].astype(bool)
        keys = z["target_time_utc"].astype(str)
        seq = z["sequence_ids"].astype(str)
    expected = (len(seq), 2, 16, 16, 16)
    if pred_log.shape != expected or persist_log.shape != expected or ref.shape != expected or vmask.shape != expected:
        raise RuntimeError("validation bank shapes do not match fixed 16x16 contract")
    if not np.isfinite(ref[vmask]).all():
        raise RuntimeError("non-finite reference under common mask")

    pred = np.expm1(pred_log)
    persistence = np.expm1(persist_log)
    kernel = gaussian_kernel()
    records, support_counts = [], {"S_to_R": 0, "persistence_R": 0}
    clipped_low = {"S_to_R": 0, "persistence_R": 0}
    clipped_high = {"S_to_R": 0, "persistence_R": 0}
    reference_clipped_high = 0
    candidates = int(vmask.sum())
    for i in range(len(seq)):
        for s, station in enumerate(("sili", "zhujia")):
            for l in range(16):
                valid = vmask[i, s, l]
                if not valid.any():
                    continue
                reference_clipped_high += int((valid & (ref[i, s, l] > q999)).sum())
                truth = np.clip(np.log1p(np.maximum(ref[i, s, l], 0.0)) / L, 0.0, 1.0)
                pair = {"S_to_R": pred[i, s, l], "persistence_R": persistence[i, s, l]}
                for method, px in pair.items():
                    clipped_low[method] += int((valid & (px < 0)).sum())
                    clipped_high[method] += int((valid & (px > q999)).sum())
                    mapped = np.clip(np.log1p(np.maximum(px, 0.0)) / L, 0.0, 1.0)
                    ss = ssim_valid(mapped, truth, valid, kernel)
                    support_counts[method] += len(ss)
                    records.append({
                        "sequence_id": seq[i], "station": station,
                        "lead_min": 15 * (l + 1), "method": method,
                        "n_valid_pixels": int(valid.sum()), "n_valid_11x11_windows": int(len(ss)),
                        "mean_ssim": float(ss.mean()) if len(ss) else None,
                    })
    usable = [r for r in records if r["mean_ssim"] is not None]
    # Keep the planned station/lead equal weighting, with windows averaged inside each cell.
    macro = {}
    for method in ("S_to_R", "persistence_R"):
        cell_scores = []
        for station in ("sili", "zhujia"):
            for lead in range(15, 241, 15):
                cells = [r["mean_ssim"] for r in usable
                         if r["method"] == method and r["station"] == station and r["lead_min"] == lead]
                if cells:
                    cell_scores.append(float(np.mean(cells)))
        macro[method] = float(np.mean(cell_scores)) if cell_scores else None
    a.output.mkdir(parents=True)
    with (a.output / "cot_ssim_by_window.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    result = {
        "state": "COMPLETE_VALIDATION_SSIM_TRAIN_SCALE_ONLY",
        "split": "train_reference_scale_and_validation_score_only",
        "test_payloads_read": 0,
        "scale": {
            "source": "CPP reference COT values with valid mask in monthly_R_station16_20261013_v1/data/pack_trainval_v1; train rows only",
            "train_supervised_examples": int(len(train_ids)),
            "train_valid_reference_pixels": int(vals.size),
            "train_reference_q999_physical_COT": q999,
            "L_log1p_q999": L,
            "quantile_method": "numpy linear",
            "train_cot_sha256": sha(cot_path), "train_mask_sha256": sha(mask_path),
        },
        "validation": {
            "bank_sha256": sha(a.bank), "n_windows": int(len(seq)),
            "target_key_shape": list(keys.shape), "n_common_cpp_valid_pixels": candidates,
            "window": 11, "gaussian_sigma": 1.5, "K1": 0.01, "K2": 0.03,
            "covariance": "population", "data_range": 1.0,
            "validity": "score only centers whose complete 11x11 support is valid; no fill",
            "supported_window_count": support_counts,
            "nominal_11x11_window_count": int(len(seq) * 2 * 16 * 36),
            "supported_window_fraction_of_nominal": {
                method: float(count / (len(seq) * 2 * 16 * 36))
                for method, count in support_counts.items()
            },
            "mean_window_then_equal_station_lead_macro_ssim": macro,
            "validation_reference_above_train_q999_pixels": reference_clipped_high,
            "prediction_clip_counts_in_common_pixels": {"below_zero": clipped_low, "above_train_q999": clipped_high},
            "n_records": len(records), "n_records_with_ssim": len(usable),
        },
        "caveat": "Only eight CPP-supported BJT target dates; CPP is a retrieval reference, not independent COT truth. SSIM is a secondary log1p-COT structure metric and does not establish deployment skill.",
    }
    (a.output / "cot_ssim_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
