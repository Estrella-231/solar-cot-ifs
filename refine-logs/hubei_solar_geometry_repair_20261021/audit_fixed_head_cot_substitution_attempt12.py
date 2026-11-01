#!/usr/bin/env python3
"""Independent FP64 recomputation from saved common-support predictions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
HORIZONS = {"all": (15, 240), "1h": (15, 60), "2h": (75, 120),
            "3h": (135, 180), "4h": (195, 240)}
ARMS = ("forecast_cot", "history_persistence", "offline_future_agri")


def rmse(rows, arm):
    e = np.asarray([float(r[arm]) - float(r["truth"]) for r in rows], dtype=np.float64)
    return float(np.sqrt(np.mean(e * e, dtype=np.float64)))


def date_ci(rows_by_station, alt, lo, hi, seed, draws=5000):
    date_sets = [{r["target_date_bjt"] for r in rows_by_station[st]
                  if lo <= int(r["lead_minutes"]) <= hi} for st in STATIONS]
    dates = sorted(set.intersection(*date_sets))
    if len(dates) < 2:
        return {"common_target_date_clusters": len(dates), "ci95_delta_rmse_wm2": [None, None]}
    di = {d: i for i, d in enumerate(dates)}
    stats = {}
    point_station = []
    for st in STATIONS:
        rs = [r for r in rows_by_station[st] if lo <= int(r["lead_minutes"]) <= hi and r["target_date_bjt"] in di]
        sums = {arm: np.zeros(len(dates), dtype=np.float64) for arm in ("forecast_cot", alt)}
        counts = np.zeros(len(dates), dtype=np.int64)
        for r in rs:
            j = di[r["target_date_bjt"]]
            counts[j] += 1
            for arm in sums:
                e = float(r[arm]) - float(r["truth"])
                sums[arm][j] += e * e
        stats[st] = sums, counts
        n = int(counts.sum())
        point_station.append(np.sqrt(sums[alt].sum(dtype=np.float64) / n) -
                             np.sqrt(sums["forecast_cot"].sum(dtype=np.float64) / n))
    rng = np.random.default_rng(seed)
    draws_idx = rng.integers(0, len(dates), size=(draws, len(dates)))
    delta = np.empty(draws, dtype=np.float64)
    for b, draw in enumerate(draws_idx):
        ds = []
        for st in STATIONS:
            sums, counts = stats[st]
            n = int(counts[draw].sum())
            ds.append(np.sqrt(sums[alt][draw].sum(dtype=np.float64) / n) -
                      np.sqrt(sums["forecast_cot"][draw].sum(dtype=np.float64) / n))
        delta[b] = float(np.mean(ds))
    return {"common_target_date_clusters": len(dates),
            "observed_equal_station_delta_on_ci_support_wm2": float(np.mean(point_station)),
            "ci95_delta_rmse_wm2": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
            "bootstrap_draws": draws, "bootstrap_seed": seed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    path = a.result_dir / "paired_predictions_common_validation.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = [(r["station"], r["init_time_bjt"], r["target_time_bjt"], r["lead_minutes"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate paired validation keys")
    if any(not all(np.isfinite(float(r[k])) for k in ("truth", *ARMS)) for r in rows):
        raise AssertionError("Nonfinite saved validation prediction")
    by_station = {st: [r for r in rows if r["station"] == st] for st in STATIONS}
    expected = json.loads((a.result_dir / "summary.json").read_text(encoding="utf-8"))
    written = list(csv.DictReader((a.result_dir / "metrics_by_station_horizon.csv").open(encoding="utf-8-sig", newline="")))
    recomputed_metrics = []
    for st in STATIONS:
        for h, (lo, hi) in HORIZONS.items():
            subset = [r for r in by_station[st] if lo <= int(r["lead_minutes"]) <= hi]
            for arm in ARMS:
                e = np.asarray([float(r[arm]) - float(r["truth"]) for r in subset], dtype=np.float64)
                m = {"station": st, "horizon": h, "arm": arm, "n": int(len(e)),
                     "rmse_wm2": float(np.sqrt(np.mean(e * e, dtype=np.float64))),
                     "mae_wm2": float(np.mean(np.abs(e), dtype=np.float64)),
                     "bias_wm2": float(np.mean(e, dtype=np.float64))}
                candidate = next(x for x in written if x["station"] == st and x["horizon"] == h and x["arm"] == arm)
                for k in ("n", "rmse_wm2", "mae_wm2", "bias_wm2"):
                    if k == "n":
                        if int(candidate[k]) != m[k]:
                            raise AssertionError(f"Metric row count mismatch {st}/{h}/{arm}")
                    elif not np.isclose(float(candidate[k]), m[k], rtol=0, atol=1e-9):
                        raise AssertionError(f"Saved metric mismatch {st}/{h}/{arm}/{k}: {candidate[k]} vs {m[k]}")
                recomputed_metrics.append(m)

    equal_station = {}
    for i, (h, (lo, hi)) in enumerate(HORIZONS.items()):
        horizon_rows = {st: [r for r in by_station[st] if lo <= int(r["lead_minutes"]) <= hi] for st in STATIONS}
        base = [rmse(horizon_rows[st], "forecast_cot") for st in STATIONS]
        for j, arm in enumerate(("history_persistence", "offline_future_agri")):
            alt = [rmse(horizon_rows[st], arm) for st in STATIONS]
            ci_expected = expected["equal_station_and_date_cluster_bootstrap"][f"{h}/{arm}"]
            ci = date_ci(by_station, arm, lo, hi, int(ci_expected["bootstrap_seed"]), int(ci_expected["bootstrap_draws"]))
            equal_station[f"{h}/{arm}"] = {
                "forecast_equal_station_rmse_wm2": float(np.mean(base)),
                "alternative_equal_station_rmse_wm2": float(np.mean(alt)),
                "delta_rmse_wm2": float(np.mean(np.asarray(alt) - np.asarray(base))),
                "relative_rmse_reduction_percent": float(100 * (np.mean(base) - np.mean(alt)) / np.mean(base)),
                **ci,
            }
            if not np.allclose(equal_station[f"{h}/{arm}"]["ci95_delta_rmse_wm2"],
                               ci_expected["ci95_delta_rmse_wm2"], rtol=0, atol=1e-10):
                raise AssertionError(f"Bootstrap mismatch at {h}/{arm}")
    audit = {"state": "INDEPENDENT_FP64_RECOMPUTE_PASS",
             "source": path.name,
             "rows": len(rows),
             "unique_station_init_target_lead_keys": len(set(keys)),
             "metrics_recomputed_from_saved_predictions": len(recomputed_metrics),
             "test_payloads_read": 0,
             "metrics": recomputed_metrics,
             "equal_station_and_date_cluster_bootstrap": equal_station}
    (a.result_dir / "independent_fp64_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": audit["state"], "rows": audit["rows"],
                      "all": {k: v for k, v in equal_station.items() if k.startswith("all/")}}, indent=2))


if __name__ == "__main__":
    main()
