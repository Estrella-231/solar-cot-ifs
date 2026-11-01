"""Re-score the frozen SimVP->R baseline with all 16 leads kept in order.

This is a CPU-only evaluation repair for the historical full-history control
validation cohort. It reads saved validation arrays and does not load/train a
network. Test samples are never opened.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def new_bucket():
    return np.zeros(10, dtype=np.float64)


def add(bucket, pred, truth, valid):
    if not valid.any():
        return
    e = pred[valid] - truth[valid]
    thick = truth[valid] >= 30
    clear = truth[valid] < 5
    bucket[0] += e.size
    bucket[1] += np.abs(e).sum()
    bucket[2] += np.square(e).sum()
    bucket[3] += e.sum()
    bucket[4] += thick.sum()
    bucket[5] += np.square(e[thick]).sum()
    bucket[6] += (pred[valid][thick] < 10).sum()
    bucket[7] += clear.sum()
    bucket[8] += (pred[valid][clear] >= 10).sum()
    bucket[9] += 1


def finalize(x):
    n, ab, sq, bias, tn, tsq, miss, cn, fp, _ = x
    return {
        "n": int(n), "mae": float(ab / n) if n else None,
        "rmse": float(np.sqrt(sq / n)) if n else None,
        "bias": float(bias / n) if n else None,
        "thick_n_cot_ge30": int(tn),
        "thick_rmse": float(np.sqrt(tsq / tn)) if tn else None,
        "thick_miss_rate_pred_lt10": float(miss / tn) if tn else None,
        "clear_n_cot_lt5": int(cn),
        "clear_false_rate_pred_ge10": float(fp / cn) if cn else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--days", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refuse to overwrite: {a.output}")
    d = a.bank / "val"
    forecast = np.load(d / "cot_local_forecast.npy", mmap_mode="r")
    target = np.load(d / "cot_local_cpp.npy", mmap_mode="r")
    mask = np.load(d / "cot_local_cpp_mask.npy", mmap_mode="r")
    motion = np.load(d / "motion_local32.npy", mmap_mode="r")
    days = np.load(a.days, allow_pickle=False)["init_day_BJT"].astype(str)
    n = len(forecast)
    assert forecast.shape == target.shape == mask.shape == (n, 2, 16, 1, 64, 64)
    assert motion.shape == (n, 2) and days.shape == (n,)
    yy, xx = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(64, dtype=np.float32), indexing="ij")
    all_b = {s: new_bucket() for s in ("patch", "station")}
    lead_b = {s: [new_bucket() for _ in range(16)] for s in ("patch", "station")}
    day_b = defaultdict(lambda: {s: new_bucket() for s in ("patch", "station")})
    day_lead_b = defaultdict(lambda: {s: [new_bucket() for _ in range(16)] for s in ("patch", "station")})
    for i in range(n):
        # Channel and lead axes are explicit: forecast is [N,station,T,C,H,W].
        f = np.expm1(np.asarray(forecast[i, 0, :, 0], dtype=np.float32))
        y = np.expm1(np.asarray(target[i, 0, :, 0], dtype=np.float32))
        m = np.asarray(mask[i, 0, :, 0] > 0.5)
        vy, vx = float(motion[i, 0]), float(motion[i, 1])
        day = str(days[i])
        for t in range(16):
            sy, sx = yy - (t + 1) * vy, xx - (t + 1) * vx
            support = (sy >= 0) & (sy <= 63) & (sx >= 0) & (sx <= 63)
            valid = m[t] & support
            add(all_b["patch"], f[t], y[t], valid)
            add(lead_b["patch"][t], f[t], y[t], valid)
            add(day_b[day]["patch"], f[t], y[t], valid)
            add(day_lead_b[day]["patch"][t], f[t], y[t], valid)
            v = bool(valid[32, 32])
            if v:
                add(all_b["station"], f[t, 32:33, 32:33], y[t, 32:33, 32:33], np.ones((1, 1), bool))
                add(lead_b["station"][t], f[t, 32:33, 32:33], y[t, 32:33, 32:33], np.ones((1, 1), bool))
                add(day_b[day]["station"], f[t, 32:33, 32:33], y[t, 32:33, 32:33], np.ones((1, 1), bool))
                add(day_lead_b[day]["station"][t], f[t, 32:33, 32:33], y[t, 32:33, 32:33], np.ones((1, 1), bool))
    a.output.mkdir(parents=True)
    summary = {
        "state": "COMPLETE_CORRECTED_SIMVP_LEAD_AXIS_VALIDATION",
        "test_used": False, "training_performed": False,
        "cohort_sequences": int(n), "station": "Sili, patch-bank station index 0; station pixel (32,32)",
        "lead_minutes": list(range(15, 241, 15)),
        "forecast_shape_contract": list(forecast.shape),
        "correction": "Preserve [sequence, lead, COT, row, col] and score each SimVP lead separately; previous aggregate used lead 1 repeated 16 times.",
        "support": "Same CPP-valid intersect constant-translation in-domain support as the existing control comparison.",
        "reference": "CPP retrieval product; not independent physical truth.",
        "overall": {k: finalize(v) for k, v in all_b.items()},
        "per_lead": {k: [finalize(v) for v in vals] for k, vals in lead_b.items()},
        "per_init_day": {day: {k: finalize(v) for k, v in scopes.items()} for day, scopes in day_b.items()},
        "per_init_day_lead": {day: {k: [finalize(v) for v in vals] for k, vals in scopes.items()}
                               for day, scopes in day_lead_b.items()},
    }
    (a.output / "corrected_simvp_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.output / "complete.json").write_text(json.dumps({
        "state": summary["state"], "test_used": False, "training_performed": False,
        "cohort_sequences": n, "forecast_shape_contract": summary["forecast_shape_contract"],
        "support": summary["support"], "corrected_script": Path(__file__).name,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"state": summary["state"], "overall": summary["overall"],
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
