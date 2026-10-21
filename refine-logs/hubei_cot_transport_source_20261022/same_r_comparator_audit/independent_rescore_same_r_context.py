#!/usr/bin/env python3
"""Independent FP64 rescore of the partial same-R validation records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STATIONS = ["Niushou", "Wangzhai", "Mizhuang"]
LEADS = list(range(15, 241, 15))
METHODS = ["cot_forecast", "persistence", "simvp_forecast_agri_to_same_r"]
THICKNESS = [("near_zero_lt1", 0.0, 1.0), ("thin_1_10", 1.0, 10.0),
             ("moderate_10_30", 10.0, 30.0), ("thick_ge30", 30.0, float("inf"))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261022)
    a = p.parse_args()
    with a.records.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 5460:
        raise ValueError(f"Unexpected station-record count: {len(rows)}")
    pair_keys = {(r["init_time_bjt"], r["target_time_bjt"], int(r["lead_minutes"])) for r in rows}
    dates = sorted({r["date"] for r in rows})
    # Each supported key has at least one station record; 12 of the potential
    # 3 x 1824 center records are absent because that station pixel is invalid.
    if len(pair_keys) != 1824:
        raise ValueError(f"Unexpected unique init/target/lead keys: {len(pair_keys)}")
    if len(dates) != 16 or sorted({int(r["lead_minutes"]) for r in rows}) != LEADS:
        raise ValueError("Unexpected validation dates or lead support")

    dmap, smap, lmap = ({v: i for i, v in enumerate(values)} for values in (dates, STATIONS, LEADS))
    shape = (len(dates), len(STATIONS), len(LEADS), len(METHODS))
    sse = np.zeros(shape, dtype=np.float64)
    sae = np.zeros(shape, dtype=np.float64)
    signed = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape[:3], dtype=np.int64)
    thickness_shape = (len(STATIONS), len(LEADS))
    thickness_stats = {name: [np.zeros(thickness_shape + (len(METHODS),), dtype=np.float64),
                              np.zeros(thickness_shape + (len(METHODS),), dtype=np.float64),
                              np.zeros(thickness_shape + (len(METHODS),), dtype=np.float64),
                              np.zeros(thickness_shape, dtype=np.int64)]
                       for name, _, _ in THICKNESS}
    lead_record_counts = {lead: 0 for lead in LEADS}

    for r in rows:
        di, si, li = dmap[r["date"]], smap[r["station"]], lmap[int(r["lead_minutes"])]
        truth = float(r["truth"])
        estimates = [float(r[m]) for m in METHODS]
        errors = np.asarray(estimates, dtype=np.float64) - truth
        sse[di, si, li] += errors * errors
        sae[di, si, li] += np.abs(errors)
        signed[di, si, li] += errors
        count[di, si, li] += 1
        lead_record_counts[int(r["lead_minutes"])] += 1
        for name, low, high in THICKNESS:
            if low <= truth < high:
                ss, aa, bb, nn = thickness_stats[name]
                ss[si, li] += errors * errors
                aa[si, li] += np.abs(errors)
                bb[si, li] += errors
                nn[si, li] += 1
                break

    n_by_cell = count.sum(axis=0)
    if not np.all(n_by_cell > 0):
        raise ValueError("At least one station x lead cell has no common-support records")
    sse_total, sae_total, signed_total = sse.sum(axis=0), sae.sum(axis=0), signed.sum(axis=0)
    rmse_cells = np.sqrt(sse_total / n_by_cell[..., None])
    mae_cells = sae_total / n_by_cell[..., None]
    bias_cells = signed_total / n_by_cell[..., None]
    overall = {
        name: {
            "equal_station_equal_lead_rmse_cot": float(rmse_cells[:, :, mi].mean()),
            "equal_station_equal_lead_mae_cot": float(mae_cells[:, :, mi].mean()),
            "equal_station_equal_lead_bias_cot": float(bias_cells[:, :, mi].mean()),
            "station_center_records": int(count.sum()),
        }
        for mi, name in enumerate(METHODS)
    }
    by_station = {
        STATIONS[si]: {name: float(rmse_cells[si, :, mi].mean()) for mi, name in enumerate(METHODS)}
        for si in range(len(STATIONS))
    }
    by_lead = {
        str(LEADS[li]): {name: float(rmse_cells[:, li, mi].mean()) for mi, name in enumerate(METHODS)}
        for li in range(len(LEADS))
    }

    rng = np.random.default_rng(a.seed)
    draws = np.stack([rng.choice(len(dates), size=len(dates), replace=True) for _ in range(a.bootstrap)])
    boot_sse = sse[draws].sum(axis=1)
    boot_n = count[draws].sum(axis=1)
    if not np.all(boot_n > 0):
        raise ValueError("A date-block bootstrap replicate has an empty station/lead cell")
    boot_rmse = np.sqrt(boot_sse / boot_n[..., None]).mean(axis=(1, 2))
    deltas = {
        "persistence_minus_cot_forecast": boot_rmse[:, METHODS.index("persistence")] - boot_rmse[:, 0],
        "simvp_forecast_agri_to_same_r_minus_cot_forecast": boot_rmse[:, 2] - boot_rmse[:, 0],
    }
    intervals = {name: {"point_delta_rmse_cot": float(
                            overall[("persistence" if name.startswith("persistence") else "simvp_forecast_agri_to_same_r")]["equal_station_equal_lead_rmse_cot"] -
                            overall["cot_forecast"]["equal_station_equal_lead_rmse_cot"]),
                        "ci95_date_block": np.quantile(v, [0.025, 0.975]).tolist(),
                        "bootstrap_mean_delta": float(v.mean())}
                 for name, v in deltas.items()}

    strata = {}
    for name, (ss, aa, bb, nn) in thickness_stats.items():
        active = nn > 0
        if not active.any():
            continue
        rm = np.sqrt(ss[active] / nn[active, None]).mean(axis=0)
        ma = (aa[active] / nn[active, None]).mean(axis=0)
        bi = (bb[active] / nn[active, None]).mean(axis=0)
        strata[name] = {
            "station_lead_cells": int(active.sum()),
            "station_center_records": int(nn.sum()),
            "equal_station_equal_lead_rmse_cot": {m: float(rm[i]) for i, m in enumerate(METHODS)},
            "equal_station_equal_lead_mae_cot": {m: float(ma[i]) for i, m in enumerate(METHODS)},
            "equal_station_equal_lead_bias_cot": {m: float(bi[i]) for i, m in enumerate(METHODS)},
        }
    result = {
        "state": "INDEPENDENT_FP64_RESCORE_VALIDATION_PARTIAL_COMMON_SUPPORT",
        "records": len(rows),
        "unique_init_target_lead_keys": len(pair_keys),
        "target_dates_bjt": dates,
        "lead_record_counts": {str(k): v for k, v in lead_record_counts.items()},
        "all_station_lead_cells_nonempty": True,
        "methods": METHODS,
        "metrics": {"overall_equal_station_equal_lead": overall,
                    "by_station_equal_lead": by_station,
                    "by_lead_equal_station": by_lead,
                    "by_cpp_retrieved_target_thickness": strata},
        "paired_date_block_intervals": intervals,
        "bootstrap_replicates": a.bootstrap,
        "bootstrap_seed": a.seed,
        "warning": "CPP is a retrieval reference; support is a 1824-key intersection, not the full 3220-key G1S score. Attempt12 SimVP is contextual, not the matched attempt13 MSE arm. Validation-only; not GHI evidence or independent test.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"]["overall_equal_station_equal_lead"], indent=2))
    print(json.dumps(result["paired_date_block_intervals"], indent=2))
    print(json.dumps(result["metrics"]["by_cpp_retrieved_target_thickness"], indent=2))


if __name__ == "__main__":
    main()
