#!/usr/bin/env python3
"""Paired FP64 comparison of attempt13 against attempt9 on saved common keys.

This is a validation-only diagnostic. It joins attempt13 saved forecasts to
the already frozen attempt9/persistence/CPP-reference records from the 1,824
key common-support audit. It does not train, inspect test/buffer data, or treat
CPP as independent physical truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STATIONS = ["Niushou", "Wangzhai", "Mizhuang"]
PIXELS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LEADS = list(range(15, 241, 15))
METHODS = ["attempt9_learned", "attempt13_persistence_residual", "persistence"]
LOG_DENOM = float(np.log(101.0))
THICKNESS = [("near_zero_lt1", 0.0, 1.0), ("thin_1_10", 1.0, 10.0),
             ("moderate_10_30", 10.0, 30.0), ("thick_ge30", 30.0, float("inf"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    with np.load(args.predictions, allow_pickle=False) as z:
        init_keys = [str(x) for x in z["init_time_bjt"].tolist()]
        pred_scaled = np.asarray(z["pred_log_scaled"], dtype=np.float64)
    if pred_scaled.shape != (len(init_keys), 16, 1, 16, 16):
        raise ValueError(f"unexpected forecast shape {pred_scaled.shape}")
    if len(set(init_keys)) != len(init_keys):
        raise ValueError("duplicate attempt13 init_time keys")
    init_index = {key: i for i, key in enumerate(init_keys)}
    if not np.isfinite(pred_scaled).all() or pred_scaled.min() < 0 or pred_scaled.max() > 1:
        raise ValueError("attempt13 scaled log-COT predictions violate [0,1]")
    # The model is bounded in scaled log1p space. Clipping only removes the
    # 2.8e-14 expm1 round-off above 100 at exactly scaled value 1.
    pred_cot = np.clip(np.expm1(pred_scaled * LOG_DENOM), 0.0, 100.0)

    with args.records.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keys = {(r["init_time_bjt"], r["target_time_bjt"], int(r["lead_minutes"])) for r in rows}
    if len(rows) != 5460 or len(keys) != 1824:
        raise ValueError(f"unexpected common support: {len(rows)} rows, {len(keys)} keys")
    if sorted({int(r["lead_minutes"]) for r in rows}) != LEADS:
        raise ValueError("common support does not cover all canonical leads")

    dates = sorted({r["date"] for r in rows})
    dmap = {d: i for i, d in enumerate(dates)}
    smap = {s: i for i, s in enumerate(STATIONS)}
    lmap = {lead: i for i, lead in enumerate(LEADS)}
    shape = (len(dates), len(STATIONS), len(LEADS), len(METHODS))
    sse = np.zeros(shape, dtype=np.float64)
    sae = np.zeros(shape, dtype=np.float64)
    bias = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape[:3], dtype=np.int64)
    thickness_sse = {name: np.zeros((len(STATIONS), len(LEADS), len(METHODS)), dtype=np.float64)
                     for name, _, _ in THICKNESS}
    thickness_sae = {name: np.zeros_like(a) for name, a in thickness_sse.items()}
    thickness_bias = {name: np.zeros_like(a) for name, a in thickness_sse.items()}
    thickness_n = {name: np.zeros((len(STATIONS), len(LEADS)), dtype=np.int64)
                   for name, _, _ in THICKNESS}

    seen = set()
    for r in rows:
        station, init_s = r["station"], r["init_time_bjt"]
        lead = int(r["lead_minutes"])
        if station not in smap or init_s not in init_index or lead not in lmap:
            raise ValueError(f"unmatched row key: {r}")
        if r["target_time_bjt"] != str(int(init_s) + lead):
            # Parse as datetime below to avoid calendar rollover pitfalls.
            from datetime import datetime, timedelta
            expected = datetime.strptime(init_s, "%Y%m%d%H%M") + timedelta(minutes=lead)
            if expected.strftime("%Y%m%d%H%M") != r["target_time_bjt"]:
                raise ValueError(f"target/init/lead mismatch: {r}")
        unique = (station, init_s, r["target_time_bjt"], lead)
        if unique in seen:
            raise ValueError(f"duplicate station forecast key: {unique}")
        seen.add(unique)
        rr, cc = PIXELS[station]
        k = lead // 15 - 1
        candidate = float(pred_cot[init_index[init_s], k, 0, rr, cc])
        truth = float(r["truth"])
        estimates = [float(r["cot_forecast"]), candidate, float(r["persistence"])]
        errors = np.asarray(estimates, dtype=np.float64) - truth
        di, si, li = dmap[r["date"]], smap[station], lmap[lead]
        sse[di, si, li] += errors * errors
        sae[di, si, li] += np.abs(errors)
        bias[di, si, li] += errors
        count[di, si, li] += 1
        for name, low, high in THICKNESS:
            if low <= truth < high:
                thickness_sse[name][si, li] += errors * errors
                thickness_sae[name][si, li] += np.abs(errors)
                thickness_bias[name][si, li] += errors
                thickness_n[name][si, li] += 1
                break
    if len(seen) != len(rows) or not np.all(count.sum(axis=0) > 0):
        raise ValueError("incomplete or duplicate common station/lead support")

    ncell = count.sum(axis=0)
    rmse_cell = np.sqrt(sse.sum(axis=0) / ncell[..., None])
    mae_cell = sae.sum(axis=0) / ncell[..., None]
    bias_cell = bias.sum(axis=0) / ncell[..., None]
    overall = {
        name: {"equal_station_equal_lead_rmse": float(rmse_cell[:, :, mi].mean()),
               "equal_station_equal_lead_mae": float(mae_cell[:, :, mi].mean()),
               "equal_station_equal_lead_bias": float(bias_cell[:, :, mi].mean())}
        for mi, name in enumerate(METHODS)
    }
    by_station = {
        station: {name: float(rmse_cell[si, :, mi].mean()) for mi, name in enumerate(METHODS)}
        for si, station in enumerate(STATIONS)
    }

    rng = np.random.default_rng(args.seed)
    draws = np.stack([rng.choice(len(dates), size=len(dates), replace=True)
                      for _ in range(args.bootstrap)])
    boot_sse = sse[draws].sum(axis=1)
    boot_n = count[draws].sum(axis=1)
    if not np.all(boot_n > 0):
        raise ValueError("bootstrap replicate has an empty station/lead cell")
    boot_rmse = np.sqrt(boot_sse / boot_n[..., None]).mean(axis=(1, 2))
    deltas = {
        "attempt9_minus_attempt13": boot_rmse[:, 0] - boot_rmse[:, 1],
        "persistence_minus_attempt13": boot_rmse[:, 2] - boot_rmse[:, 1],
    }
    paired = {}
    for name, samples in deltas.items():
        point = (overall["attempt9_learned"]["equal_station_equal_lead_rmse"] -
                 overall["attempt13_persistence_residual"]["equal_station_equal_lead_rmse"]
                 if name.startswith("attempt9") else
                 overall["persistence"]["equal_station_equal_lead_rmse"] -
                 overall["attempt13_persistence_residual"]["equal_station_equal_lead_rmse"])
        paired[name] = {"point_delta_rmse_favor_attempt13_if_positive": float(point),
                        "date_block_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                        "bootstrap_replicates": args.bootstrap}

    thickness = {}
    for name, _, _ in THICKNESS:
        n = thickness_n[name]
        active = n > 0
        if not active.any():
            continue
        thickness[name] = {
            "station_lead_cells": int(active.sum()),
            "rmse": {m: float(np.sqrt(thickness_sse[name][active, i] / n[active]).mean())
                     for i, m in enumerate(METHODS)},
            "bias": {m: float((thickness_bias[name][active, i] / n[active]).mean())
                     for i, m in enumerate(METHODS)},
        }
    result = {
        "state": "VALIDATION_ONLY_PAIRED_FP64_RESCORE_ATTEMPT13_VS_ATTEMPT9",
        "reference_warning": "CPP is a retrieval reference, not independent cloud truth.",
        "scope": "1,824-key common support previously used by the contextual audit; not an independent test.",
        "records": len(rows), "unique_init_target_lead_keys": len(keys),
        "dates_bjt": dates, "prediction_init_keys": len(init_keys),
        "methods": METHODS, "overall": overall, "by_station_equal_lead_rmse": by_station,
        "paired_date_block_intervals": paired, "by_cpp_retrieved_thickness": thickness,
        "bootstrap_seed": args.seed, "test_payloads_read": 0, "buffer_payloads_read": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "paired": paired, "thickness": thickness}, indent=2))


if __name__ == "__main__":
    main()
