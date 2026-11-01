#!/usr/bin/env python3
"""Recompute T18 AC-A CI with shared-union and station-separate date resampling.

Reads only saved T18 validation predictions and their paired validation keys.
No training, server access, or test payloads are used.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ARMS = ("A", "AC", "AP", "AL")
STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)
CONTRASTS = (("AL", "A"), ("AL", "AC"), ("AL", "AP"), ("AC", "A"))
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(y: np.ndarray, p: np.ndarray, multiplicity: np.ndarray | None = None) -> float:
    err = np.asarray(p, np.float64) - np.asarray(y, np.float64)
    if multiplicity is None:
        return float(np.sqrt(np.mean(err * err, dtype=np.float64)))
    w = np.asarray(multiplicity, np.float64)
    den = float(w.sum())
    if den <= 0:
        raise ValueError("bootstrap draw has zero support for a station")
    return float(np.sqrt(np.sum(w * err * err, dtype=np.float64) / den))


def read_data(audit_dir: Path):
    z = np.load(audit_dir / "paired_predictions_fp64.npz", allow_pickle=False)
    with (audit_dir / "paired_val_keys.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.asarray(z["ghi_truth_wm2"], np.float64)
    stations = np.asarray(z["station"]).astype(str)
    dates = np.asarray([r["BJT_date"] for r in rows])
    if len(y) != len(rows) or not np.array_equal(stations, np.asarray([r["station"] for r in rows])):
        raise ValueError("prediction/key alignment mismatch")
    preds = {(arm, seed): np.asarray(z[f"{arm}_seed{seed}_ghi_wm2"], np.float64)
             for arm in ARMS for seed in SEEDS}
    if any(v.shape != y.shape or not np.isfinite(v).all() for v in preds.values()):
        raise ValueError("prediction shape or finite-value check failed")
    return y, stations, dates, preds


def point_estimate(y, stations, preds):
    by_arm_station = {}
    for arm in ARMS:
        by_arm_station[arm] = {}
        for station in STATIONS:
            ix = stations == station
            by_arm_station[arm][station] = float(np.mean([
                rmse(y[ix], preds[(arm, seed)][ix]) for seed in SEEDS
            ]))
    return {f"{cand}-{base}": float(np.mean([
        by_arm_station[cand][s] - by_arm_station[base][s] for s in STATIONS
    ])) for cand, base in CONTRASTS}


def daily_sse_stats(y, stations, dates, preds):
    """Preaggregate row counts and FP64 squared-error sums by station/date."""
    date_sets = {s: sorted(set(dates[stations == s].tolist())) for s in STATIONS}
    stats = {}
    for station in STATIONS:
        stats[station] = {}
        for arm in ARMS:
            stats[station][arm] = {}
            for seed_id in SEEDS:
                n_by_date, sse_by_date = [], []
                p = preds[(arm, seed_id)]
                for day in date_sets[station]:
                    ix = (stations == station) & (dates == day)
                    err = p[ix] - y[ix]
                    n_by_date.append(int(ix.sum()))
                    sse_by_date.append(float(np.sum(err * err, dtype=np.float64)))
                stats[station][arm][seed_id] = (
                    np.asarray(n_by_date, dtype=np.float64),
                    np.asarray(sse_by_date, dtype=np.float64),
                )
    return date_sets, stats


def rmse_from_daily(stats, station, arm, seed_id, multiplicities):
    n, sse = stats[station][arm][seed_id]
    denom = float(np.dot(multiplicities, n))
    if denom <= 0:
        raise ValueError("bootstrap draw has zero support for a station")
    return float(np.sqrt(np.dot(multiplicities, sse) / denom))


def union_date_bootstrap(date_sets, stats, seed):
    all_dates = sorted(set(d for s in STATIONS for d in date_sets[s]))
    rng = np.random.default_rng(seed)
    values = {f"{c}-{b}": [] for c, b in CONTRASTS}
    for _ in range(N_BOOT):
        draw = rng.choice(all_dates, size=len(all_dates), replace=True)
        multiplicity = Counter(draw.tolist())
        weights = {s: np.asarray([multiplicity.get(d, 0) for d in date_sets[s]], dtype=np.float64)
                   for s in STATIONS}
        agg = {key: [] for key in values}
        for seed_id in SEEDS:
            for cand, base in CONTRASTS:
                station_delta = []
                for station in STATIONS:
                    d = rmse_from_daily(stats, station, cand, seed_id, weights[station]) - rmse_from_daily(
                        stats, station, base, seed_id, weights[station]
                    )
                    station_delta.append(d)
                agg[f"{cand}-{base}"].append(float(np.mean(station_delta)))
        for key in values:
            values[key].append(float(np.mean(agg[key])))
    return {k: [float(x) for x in np.quantile(v, [0.025, 0.975])] for k, v in values.items()}


def station_separate_bootstrap(date_sets, stats, seed):
    rng = np.random.default_rng(seed)
    values = {f"{c}-{b}": [] for c, b in CONTRASTS}
    for _ in range(N_BOOT):
        draws = {s: rng.choice(date_sets[s], size=len(date_sets[s]), replace=True) for s in STATIONS}
        weights = {s: np.asarray([Counter(draws[s].tolist()).get(d, 0) for d in date_sets[s]], dtype=np.float64)
                   for s in STATIONS}
        station_scores = {s: {} for s in STATIONS}
        for station in STATIONS:
            for arm in ARMS:
                station_scores[station][arm] = float(np.mean([
                    rmse_from_daily(stats, station, arm, seed_id, weights[station]) for seed_id in SEEDS
                ]))
        for cand, base in CONTRASTS:
            values[f"{cand}-{base}"].append(float(np.mean([
                station_scores[s][cand] - station_scores[s][base] for s in STATIONS
            ])))
    return {k: [float(x) for x in np.quantile(v, [0.025, 0.975])] for k, v in values.items()}


def main():
    here = Path(__file__).resolve().parent
    project_root = here.parents[3]
    audit_dir = (project_root / "experiments/monthly_R_station16_20261013/results/"
                 "r_al_formal_t18_20261014_v1/fp64_audit_v3")
    y, stations, dates, preds = read_data(audit_dir)
    point = point_estimate(y, stations, preds)
    date_sets, daily_stats = daily_sse_stats(y, stations, dates, preds)
    union_ci = union_date_bootstrap(date_sets, daily_stats, seed=20261013)
    separate_ci = station_separate_bootstrap(date_sets, daily_stats, seed=20261014)
    report = {
        "state": "PASS_T18_BOOTSTRAP_METHOD_RECONCILIATION",
        "scope": "validation-only saved T18 predictions; no test payloads",
        "test_payloads_read": 0,
        "n_rows": int(len(y)),
        "n_union_bjt_dates": int(len(set(dates.tolist()))),
        "station_bjt_date_counts": {s: int(len(set(dates[stations == s].tolist()))) for s in STATIONS},
        "point_difference_wm2": point,
        "union_date_shared_pairing_95ci_wm2_seed_20261013": union_ci,
        "station_separate_date_95ci_wm2_seed_20261014": separate_ci,
        "input_sha256": {
            "paired_predictions_fp64.npz": sha256(audit_dir / "paired_predictions_fp64.npz"),
            "paired_val_keys.csv": sha256(audit_dir / "paired_val_keys.csv"),
            "primary_audit_ghi_heads_three_seed.py": sha256(project_root / "experiments/monthly_R_station16_20261013_v1/audit_ghi_heads_three_seed.py"),
            "T18_verify_t18_r_al_saved_predictions.py": sha256(project_root / "experiments/monthly_R_station16_20261013/verify_t18_r_al_saved_predictions.py")
        },
        "interpretation": "Both intervals are paired and use the same saved predictions and point estimand; the primary interval shares sampled BJT target dates across stations, while T18 sensitivity samples each station's dates independently. Use the preregistered shared-date interval for the primary AC-A claim. The other interval is a station-separate resampling sensitivity, not a prediction or metric error."
    }
    out = here / "T18_BOOTSTRAP_METHOD_RECONCILIATION_20261014.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
