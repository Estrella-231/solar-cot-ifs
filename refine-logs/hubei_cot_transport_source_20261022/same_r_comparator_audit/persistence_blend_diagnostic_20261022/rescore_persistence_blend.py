#!/usr/bin/env python3
"""Post-hoc fixed 50/50 forecast/persistence blend diagnostic.

This is a validation-only diagnostic, not model selection or a gate result.
It uses the already saved exact common-support records and does not tune alpha.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STATIONS = ["Niushou", "Wangzhai", "Mizhuang"]
LEADS = list(range(15, 241, 15))
METHODS = ["cot_forecast", "persistence", "blend_50_50"]
STRATA = [("near_zero_lt1", 0., 1.), ("thin_1_10", 1., 10.),
          ("moderate_10_30", 10., 30.), ("thick_ge30", 30., float("inf"))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261023)
    a = p.parse_args()
    with a.records.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"station", "date", "init_time_bjt", "target_time_bjt",
                "lead_minutes", "truth", "cot_forecast", "persistence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input columns are missing")
    keys = [(r["station"], r["init_time_bjt"], r["target_time_bjt"],
             int(r["lead_minutes"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate station/init/target/lead rows")

    dates = sorted({r["date"] for r in rows})
    if sorted({r["station"] for r in rows}) != sorted(STATIONS):
        raise ValueError("Unexpected station support")
    if sorted({int(r["lead_minutes"]) for r in rows}) != LEADS:
        raise ValueError("Unexpected lead support")
    dmap = {v: i for i, v in enumerate(dates)}
    smap = {v: i for i, v in enumerate(STATIONS)}
    lmap = {v: i for i, v in enumerate(LEADS)}

    # Per-date x station x lead x method sums, retaining the original support.
    shape = (len(dates), len(STATIONS), len(LEADS), len(METHODS))
    sse = np.zeros(shape, dtype=np.float64)
    sae = np.zeros(shape, dtype=np.float64)
    signed = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape[:3], dtype=np.int64)
    strata = {name: [np.zeros(shape[1:], dtype=np.float64),
                     np.zeros(shape[1:], dtype=np.float64),
                     np.zeros(shape[1:], dtype=np.float64),
                     np.zeros(shape[1:3], dtype=np.int64)]
              for name, _, _ in STRATA}

    for r in rows:
        di, si = dmap[r["date"]], smap[r["station"]]
        li = lmap[int(r["lead_minutes"])]
        y = float(r["truth"])
        f = float(r["cot_forecast"])
        persistence = float(r["persistence"])
        estimates = np.asarray([f, persistence, .5 * f + .5 * persistence], dtype=np.float64)
        err = estimates - y
        sse[di, si, li] += err * err
        sae[di, si, li] += np.abs(err)
        signed[di, si, li] += err
        count[di, si, li] += 1
        for name, lo, hi in STRATA:
            if lo <= y < hi:
                ss, aa, bb, nn = strata[name]
                ss[si, li] += err * err
                aa[si, li] += np.abs(err)
                bb[si, li] += err
                nn[si, li] += 1
                break

    cell_count = count.sum(axis=0)
    if not np.all(cell_count > 0):
        raise ValueError("Empty station/lead cell on the common support")
    rmse_cells = np.sqrt(sse.sum(axis=0) / cell_count[..., None])
    mae_cells = sae.sum(axis=0) / cell_count[..., None]
    bias_cells = signed.sum(axis=0) / cell_count[..., None]
    overall = {
        m: {"equal_station_equal_lead_rmse_cot": float(rmse_cells[:, :, i].mean()),
            "equal_station_equal_lead_mae_cot": float(mae_cells[:, :, i].mean()),
            "equal_station_equal_lead_bias_cot": float(bias_cells[:, :, i].mean())}
        for i, m in enumerate(METHODS)
    }
    by_lead = {str(lead): {m: float(rmse_cells[:, li, i].mean())
                           for i, m in enumerate(METHODS)}
               for li, lead in enumerate(LEADS)}
    thickness = {}
    for name, (ss, aa, bb, nn) in strata.items():
        active = nn > 0
        thickness[name] = {
            "records": int(nn.sum()),
            "station_lead_cells": int(active.sum()),
            "equal_station_equal_lead_rmse_cot": {
                m: float(np.sqrt(ss[active, i] / nn[active]).mean())
                for i, m in enumerate(METHODS)},
            "equal_station_equal_lead_bias_cot": {
                m: float((bb[active, i] / nn[active]).mean())
                for i, m in enumerate(METHODS)},
        }

    rng = np.random.default_rng(a.seed)
    draws = rng.integers(0, len(dates), size=(a.bootstrap, len(dates)))
    bsse = sse[draws].sum(axis=1)
    bn = count[draws].sum(axis=1)
    if not np.all(bn > 0):
        raise ValueError("Bootstrap produced empty station/lead support")
    brmse = np.sqrt(bsse / bn[..., None]).mean(axis=(1, 2))
    pairs = {"blend_minus_forecast": (2, 0), "blend_minus_persistence": (2, 1)}
    intervals = {}
    for label, (candidate, reference) in pairs.items():
        delta = brmse[:, candidate] - brmse[:, reference]
        point = overall[METHODS[candidate]]["equal_station_equal_lead_rmse_cot"] - \
                overall[METHODS[reference]]["equal_station_equal_lead_rmse_cot"]
        intervals[label] = {"point_delta_rmse_cot": point,
                            "ci95_date_block": np.quantile(delta, [.025, .975]).tolist()}

    result = {
        "state": "POSTHOC_FIXED_BLEND_VALIDATION_DIAGNOSTIC_NOT_MODEL_SELECTION",
        "alpha_forecast": 0.5,
        "records": len(rows),
        "unique_station_init_target_lead_keys": len(set(keys)),
        "target_dates_bjt": dates,
        "common_support_as_input": True,
        "metrics": {"overall_equal_station_equal_lead": overall,
                    "by_lead": by_lead,
                    "by_cpp_retrieved_target_thickness": thickness},
        "paired_date_block_intervals": intervals,
        "bootstrap_replicates": a.bootstrap,
        "bootstrap_seed": a.seed,
        "warning": "Single fixed 50/50 arithmetic blend on previously evaluated validation data. No alpha search, no checkpoint selection, not independent confirmation, not a GHI result, and CPP is a retrieval reference rather than independent truth.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "thickness": thickness,
                      "intervals": intervals}, indent=2))


if __name__ == "__main__":
    main()
