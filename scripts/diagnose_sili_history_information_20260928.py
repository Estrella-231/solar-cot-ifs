"""Cheap held-out probe for incremental information in the full Sili COT history.

This is a fixed, regularized linear information probe on the existing train/val
bank, not a replacement for the parameter-matched neural experiment. It compares
recent two-frame COT (the current head's temporal input) against the same
predictors augmented with the six older COT frames. CPP remains a retrieval
reference; test data are never opened.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def sample_coordinates():
    # A fixed 16x16 lattice over the local 64x64 patch plus the exact station pixel.
    q = np.arange(2, 64, 4)
    yy, xx = np.meshgrid(q, q, indexing="ij")
    coords = list(zip(yy.ravel().tolist(), xx.ravel().tolist()))
    coords.append((32, 32))
    return np.asarray(coords, dtype=np.int64)


def load_features(bank, split, station, seq_count=512):
    root = Path(bank) / split
    names = ("cot_local_history", "cot_local_forecast", "cot_local_cpp",
             "cot_local_cpp_mask", "motion_local32")
    arrays = {k: np.load(root / (k + ".npy"), mmap_mode="r") for k in names}
    coords = sample_coordinates()
    cy, cx = coords.T
    nlead = 16
    all_x_recent, all_x_full, all_y, all_m, all_sup, all_station = [], [], [], [], [], []
    lead_axis = np.linspace(1 / 16, 1, 16, dtype=np.float32)
    for i in range(seq_count):
        # Stored fields are log1p(COT), with station as the second axis.
        hist = np.asarray(arrays["cot_local_history"][i, station, :, 0, cy, cx], dtype=np.float32).T
        # NumPy moves paired advanced pixel indices ahead of the sliced time
        # axis here; transpose explicitly to [lead, pixel] for every field.
        fc = np.asarray(arrays["cot_local_forecast"][i, station, :, 0, cy, cx], dtype=np.float32).T
        target = np.asarray(arrays["cot_local_cpp"][i, station, :, 0, cy, cx], dtype=np.float32).T
        mask = np.asarray(arrays["cot_local_cpp_mask"][i, station, :, 0, cy, cx] > .5).T
        motion = np.asarray(arrays["motion_local32"][i], dtype=np.float32)
        lead = np.broadcast_to(lead_axis[:, None], (nlead, len(coords)))
        dy = np.full_like(lead, motion[0] / 4.0)
        dx = np.full_like(lead, motion[1] / 4.0)

        # Use the same forecast, motion, lead, final COT and last-step change in both arms.
        recent = np.stack((fc, np.broadcast_to(hist[-1], fc.shape),
                           np.broadcast_to(hist[-1] - hist[-2], fc.shape),
                           lead, dy, dx), axis=-1)
        older = np.broadcast_to(hist[:6][None, :, :], (nlead, 6, len(coords)))
        full = np.concatenate((recent, np.moveaxis(older, 1, -1)), axis=-1)

        # Match the frozen transport comparison's source-domain support at each lead.
        sup = np.empty_like(mask)
        for k in range(1, nlead + 1):
            sy = cy - k * motion[0]
            sx = cx - k * motion[1]
            sup[k - 1] = (sy >= 0) & (sy <= 63) & (sx >= 0) & (sx <= 63)
        common = mask & sup & np.isfinite(target) & np.isfinite(recent).all(-1) & np.isfinite(full).all(-1)
        all_x_recent.append(recent.reshape(-1, recent.shape[-1]))
        all_x_full.append(full.reshape(-1, full.shape[-1]))
        all_y.append(target.reshape(-1))
        all_m.append(common.reshape(-1))
        all_sup.append(sup.reshape(-1))
        all_station.append(np.broadcast_to(np.all(coords == (32, 32), axis=1)[None, :], (nlead, len(coords))).reshape(-1))
    return (np.concatenate(all_x_recent), np.concatenate(all_x_full), np.concatenate(all_y),
            np.concatenate(all_m), np.concatenate(all_sup), np.concatenate(all_station),
            np.tile(np.repeat(np.arange(1, 17), len(coords)), seq_count))


def fit_ridge(train_x, train_y, train_mask, alpha=0.01):
    x = train_x[train_mask].astype(np.float64, copy=False)
    y = train_y[train_mask].astype(np.float64, copy=False)
    cot = np.expm1(y)
    w = 1.0 + 2.0 * (cot >= 30.0) + (cot < 5.0)
    sw = w.sum()
    mu = (x * w[:, None]).sum(axis=0) / sw
    var = ((x - mu) ** 2 * w[:, None]).sum(axis=0) / sw
    sd = np.sqrt(np.maximum(var, 1e-8))
    z = (x - mu) / sd
    ymu = (y * w).sum() / sw
    ztz = (z.T * w) @ z / sw
    zty = (z.T @ (w * (y - ymu))) / sw
    coef = np.linalg.solve(ztz + alpha * np.eye(ztz.shape[0]), zty)
    return {"mu": mu, "sd": sd, "ymu": ymu, "coef": coef, "alpha": alpha}


def predict(model, x):
    return model["ymu"] + ((x.astype(np.float64) - model["mu"]) / model["sd"]) @ model["coef"]


def scores(ylog, predlog, valid):
    y = np.clip(np.expm1(ylog[valid]), 0, 100)
    p = np.clip(np.expm1(predlog[valid]), 0, 100)
    bins = ((0, 5), (5, 10), (10, 30), (30, 60), (60, 100.0001))
    out = {"n": int(valid.sum()), "rmse": float(np.sqrt(np.mean((p-y)**2))),
           "mae": float(np.mean(np.abs(p-y))), "bias": float(np.mean(p-y))}
    out["bins"] = {}
    for lo, hi in bins:
        q = (y >= lo) & (y < hi)
        out["bins"][f"{lo:g}-{hi:g}"] = {
            "n": int(q.sum()), "rmse": float(np.sqrt(np.mean((p[q]-y[q])**2))) if q.any() else None,
            "bias": float(np.mean(p[q]-y[q])) if q.any() else None,
            "miss_lt10_rate": float(np.mean(p[q] < 10)) if lo >= 30 and q.any() else None,
        }
    thick = y >= 30
    clear = y < 5
    tp = np.sum((p >= 30) & thick)
    fp = np.sum((p >= 30) & ~thick)
    fn = np.sum((p < 30) & thick)
    out["thick_rmse"] = float(np.sqrt(np.mean((p[thick]-y[thick])**2))) if thick.any() else None
    out["thick_miss_lt10_rate"] = float(np.mean(p[thick] < 10)) if thick.any() else None
    out["thick_iou30"] = float(tp / (tp+fp+fn)) if tp+fp+fn else None
    out["clear_false10_rate"] = float(np.mean(p[clear] >= 10)) if clear.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--station", type=int, default=0, choices=(0, 1))
    ap.add_argument("--ridge-alpha", type=float, default=.01)
    a = ap.parse_args()
    meta = json.loads((a.bank / "complete.json").read_text())
    if meta.get("state") != "COMPLETE_SILI_COT_DYNAMICS_BANK" or meta.get("test_used") is not False:
        raise RuntimeError("Unexpected or test-contaminated bank")
    a.output.mkdir(parents=True, exist_ok=False)
    tr = load_features(a.bank, "train", a.station)
    va = load_features(a.bank, "val", a.station)
    results = {}
    for name, ti, vi in (("recent2", 0, 0), ("full8", 1, 1)):
        tx, fx, ty, tm = tr[ti], va[vi], tr[2], tr[3]
        model = fit_ridge(tx, ty, tm, a.ridge_alpha)
        pred = predict(model, fx)
        valid = va[3]
        station = va[5]
        lead = va[6]
        results[name] = {"all": scores(va[2], pred, valid),
                         "station_pixel": scores(va[2], pred, valid & station),
                         "lead": {str(k): scores(va[2], pred, valid & (lead == k)) for k in range(1, 17)}}
        np.save(a.output / f"{name}_validation_prediction_logcot.npy", pred.astype(np.float32))
        np.save(a.output / f"{name}_validation_common_mask.npy", valid)
        np.save(a.output / f"{name}_validation_target_logcot.npy", va[2].astype(np.float32))
    report = {
        "state": "COMPLETE_SILI_HISTORY_INFORMATION_PROBE",
        "test_used": False,
        "bank": str(a.bank),
        "station_index": a.station,
        "spatial_sample": "fixed 16x16 lattice of local64 plus exact station pixel; same coordinates in both splits",
        "common_mask": "CPP-valid intersected with source-domain transport support; same rows for both arms",
        "recent2_features": "forecast COT, last historical COT, last-minus-penultimate COT, lead, local motion dy/dx",
        "full8_features": "same recent2 features plus the six older historical COT frames",
        "estimator": {"type": "weighted linear ridge", "target": "log1p CPP-reference COT", "weights": "1 + 2 if COT>=30 + 1 if COT<5", "alpha": a.ridge_alpha},
        "limitations": ["cheap linear probe only; does not establish that a matched neural head will improve", "validation samples overlap in time", "CPP is a satellite retrieval reference, not independent cloud truth", "diagnostic sub-samples a fixed spatial lattice"],
        "results": results,
    }
    (a.output / "probe_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"all": v["all"], "station_pixel": v["station_pixel"]} for k, v in results.items()}, indent=2))
    print("SILI_HISTORY_INFORMATION_PROBE_COMPLETE")


if __name__ == "__main__":
    main()
