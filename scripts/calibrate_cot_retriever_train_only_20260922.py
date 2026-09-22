"""Fit a monotone COT output calibration on train only; evaluate validation.

This is an independent R+calibration candidate. It never opens test data or
fits any part of the mapping on validation labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_cot_repaired import COTUNet


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pava(means, weights):
    """Weighted pool-adjacent-violators fit, preserving increasing COT."""
    blocks = []
    for index, (mean, weight) in enumerate(zip(means, weights)):
        if weight <= 0:
            continue
        blocks.append([index, index, float(mean * weight), float(weight)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    out = np.full(len(means), np.nan, dtype=np.float64)
    for first, last, value_sum, weight_sum in blocks:
        out[first:last + 1] = value_sum / weight_sum
    valid = np.flatnonzero(np.isfinite(out))
    if len(valid) < 2:
        raise RuntimeError("too few occupied COT calibration bins")
    return np.interp(np.arange(len(out)), valid, out[valid])


@torch.no_grad()
def infer(model, raw, mean, std, indices, batch):
    output = np.empty((len(indices), 1, 16, 16), dtype=np.float32)
    imputed = 0
    for start in range(0, len(indices), batch):
        ids = indices[start:start + batch]
        x = (np.asarray(raw[ids], dtype=np.float32) - mean) / std
        bad = ~np.isfinite(x)
        imputed += int(bad.sum())
        x[bad] = 0
        prediction = model(torch.from_numpy(np.ascontiguousarray(x)).cuda()).float()
        if not torch.isfinite(prediction).all() or (prediction < 0).any():
            raise RuntimeError("invalid R output")
        output[start:start + len(ids)] = torch.expm1(prediction).cpu().numpy()
    return output, imputed


def metrics(pred, reference, valid):
    err = (pred - reference)[valid].astype(np.float64)
    return {"pixels": int(valid.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite COT calibration output")
    if not torch.cuda.is_available():
        raise RuntimeError("PBS GPU not visible")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    pack = root / "data/cot_repaired_pack_20260907_v1"
    config = json.loads((root / "configs/r_frozen_repaired_seed42.json").read_text())
    if sha(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise RuntimeError("R checkpoint hash drift")
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    norm = json.loads((pack / "norm.json").read_text())
    if norm != checkpoint["metadata"]["norm"] or norm["test_used"] is not False:
        raise RuntimeError("R train normalization mismatch")
    with (pack / "rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18514 or set(r["split"] for r in rows) != {"train", "validation"}:
        raise RuntimeError("pack split drift")
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"], dtype=np.int64)
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "validation"], dtype=np.int64)
    if len(train) != 11978 or len(val) != 6536:
        raise RuntimeError("pack row-count drift")
    raw = np.load(pack / "x_raw.npy", mmap_mode="r")
    truth = np.load(pack / "target.npy", mmap_mode="r")
    mask = np.load(pack / "mask.npy", mmap_mode="r")
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    model = COTUNet().float().cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    train_pred, train_imputed = infer(model, raw, mean, std, train, args.batch)
    val_pred, val_imputed = infer(model, raw, mean, std, val, args.batch)
    train_true = np.asarray(truth[train], dtype=np.float32) * np.float32(100)
    val_true = np.asarray(truth[val], dtype=np.float32) * np.float32(100)
    train_mask = np.asarray(mask[train], dtype=bool)
    val_mask = np.asarray(mask[val], dtype=bool)
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("empty reference mask")
    edges = np.linspace(0, np.log1p(100), 65)
    train_pred_valid = train_pred[train_mask].astype(np.float64)
    train_true_valid = train_true[train_mask].astype(np.float64)
    bindex = np.clip(np.searchsorted(edges, np.log1p(train_pred_valid), side="right") - 1, 0, 63)
    counts = np.bincount(bindex, minlength=64).astype(np.float64)
    target_sum = np.bincount(bindex, weights=train_true_valid, minlength=64)
    observed_mean = np.divide(target_sum, counts, out=np.zeros_like(target_sum), where=counts > 0)
    fitted = pava(observed_mean, counts)
    centers = np.expm1((edges[:-1] + edges[1:]) / 2)
    calibrated = np.interp(val_pred.ravel(), centers, fitted,
                           left=fitted[0], right=fitted[-1]).reshape(val_pred.shape).astype(np.float32)
    center = np.zeros((1, 1, 16, 16), dtype=bool)
    center[:, :, 6:11, 6:11] = True
    strata = {"all": val_mask, "cot_ge30": val_mask & (val_true >= 30),
              "station_5x5": val_mask & center, "clear_lt0p1": val_mask & (val_true < 0.1)}
    baseline = {name: metrics(val_pred, val_true, choice) for name, choice in strata.items()}
    candidate = {name: metrics(calibrated, val_true, choice) for name, choice in strata.items()}
    output.mkdir(parents=True)
    np.savez_compressed(output / "validation_predictions.npz", row_indices=val,
                        base_cot=val_pred, calibrated_cot=calibrated,
                        reference_cot=val_true, mask=val_mask)
    result = {"state": "COMPLETE_TRAIN_ONLY_MONOTONE_CALIBRATION_VALIDATION",
              "test_used": False, "base_checkpoint_sha256": config["checkpoint_sha256"],
              "source_sha256": sha(__file__), "fit_rows": len(train), "validation_rows": len(val),
              "train_nonfinite_imputed": train_imputed, "validation_nonfinite_imputed": val_imputed,
              "bin_definition": "64 uniform bins of log1p(predicted physical COT) from 0 to 100",
              "calibration": "weighted isotonic bin means, train pixels only, linear interpolation",
              "bin_centers_cot": centers.tolist(), "bin_pixel_counts": counts.astype(int).tolist(),
              "calibrated_bin_values": fitted.tolist(),
              "baseline": baseline, "candidate": candidate,
              "delta_rmse_candidate_minus_baseline": {name: candidate[name]["rmse"] - baseline[name]["rmse"]
                                                       for name in strata},
              "note": "Scalar output calibration cannot restore missing/misplaced cloud structures."}
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": result["state"], "baseline": baseline,
                      "candidate": candidate, "delta": result["delta_rmse_candidate_minus_baseline"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
