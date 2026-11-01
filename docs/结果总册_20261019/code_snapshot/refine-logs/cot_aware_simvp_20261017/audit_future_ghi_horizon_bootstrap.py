#!/usr/bin/env python3
"""Date-block paired GHI bootstrap by forecast horizon for the A/AH/AC pilot.

Reads only saved validation prediction bundles. The same BJT target-date draw is
used for both stations, all seeds, all arms and every horizon, preserving the
paired design while preventing overlapping forecast instances from acting as
independent bootstrap units.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
HORIZONS = {
    "all_15_240": (15, 240),
    "short_15_60": (15, 60),
    "middle_75_120": (75, 120),
    "long_135_240": (135, 240),
}
ALL_COMPARISONS = (("AC", "AH"), ("AC", "A"), ("AH", "A"),
                   ("AC50", "AH"), ("AC50", "AC"), ("AC50", "A"))


def _load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        pred = np.asarray(z["prediction_ghi_wm2"], dtype=np.float64)
        truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
        keys = [json.loads(str(x)) for x in z["keys"]]
    if pred.ndim != 1 or pred.shape != truth.shape or len(keys) != len(pred):
        raise ValueError(f"invalid saved prediction bundle: {path}")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError(f"nonfinite saved prediction: {path}")
    canon = [(k["station"], k["init_time_utc"], k["target_time_utc"],
              int(k["lead_minutes"]), int(k["ghi_row_index"])) for k in keys]
    if len(set(canon)) != len(canon):
        raise ValueError(f"duplicate keys: {path}")
    days = np.asarray([
        (datetime.strptime(k[2], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d")
        for k in canon
    ])
    leads = np.asarray([k[3] for k in canon], dtype=np.int16)
    return {"pred": pred, "truth": truth, "keys": canon, "days": days, "leads": leads}


def summarize(errors, day_ids, lead, days, stations, seeds, arms, comparisons, resamples):
    """Return equal-station, seed-averaged RMSE deltas and date-block CIs."""
    result = {}
    n_days = len(days)
    for hname, (lo, hi) in HORIZONS.items():
        groups = {}
        points = {s: {a: [] for a in arms} for s in stations}
        for seed in seeds:
            for station in stations:
                mask = (lead[(seed, station)] >= lo) & (lead[(seed, station)] <= hi)
                if not mask.any():
                    raise ValueError(f"empty horizon {hname} for seed={seed}, station={station}")
                ids = day_ids[(seed, station)][mask]
                for arm in arms:
                    err = errors[(seed, station, arm)][mask]
                    sq = np.bincount(ids, weights=np.square(err), minlength=n_days).astype(np.float64)
                    count = np.bincount(ids, minlength=n_days).astype(np.float64)
                    groups[(seed, station, arm)] = (sq, count)
                    points[station][arm].append(float(np.sqrt(np.mean(np.square(err)))))

        station_point = {s: {a: float(np.mean(points[s][a])) for a in arms} for s in stations}
        arm_point = {a: float(np.mean([station_point[s][a] for s in stations])) for a in arms}
        boot = {f"{l}_minus_{r}": np.empty(len(resamples), dtype=np.float64) for l, r in comparisons}
        for bi, sampled in enumerate(resamples):
            day_mult = np.bincount(sampled, minlength=n_days).astype(np.float64)
            arm_scores = {}
            for arm in arms:
                station_scores = []
                for station in stations:
                    seed_scores = []
                    for seed in seeds:
                        sq, count = groups[(seed, station, arm)]
                        den = float(day_mult @ count)
                        if den <= 0:
                            raise ValueError(f"bootstrap draw has no samples for {hname}/{station}")
                        seed_scores.append(float(np.sqrt((day_mult @ sq) / den)))
                    station_scores.append(float(np.mean(seed_scores)))
                arm_scores[arm] = float(np.mean(station_scores))
            for left, right in comparisons:
                boot[f"{left}_minus_{right}"][bi] = arm_scores[left] - arm_scores[right]
        result[hname] = {
            "lead_minutes": [lo, hi],
            "equal_station_mean_seed_rmse_wm2": arm_point,
            "per_station_mean_seed_rmse_wm2": station_point,
            "delta_rmse_wm2": {f"{l}_minus_{r}": arm_point[l] - arm_point[r]
                               for l, r in comparisons},
            "date_block_bootstrap_95pct_ci_wm2": {
                name: np.quantile(values, [0.025, 0.975]).tolist() for name, values in boot.items()
            },
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--arms", nargs="+", choices=("A", "AH", "AC", "AC50"), default=["A", "AH", "AC"])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261019)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.bootstrap < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    arms = tuple(a.arms)
    if len(set(arms)) != len(arms) or not {"A", "AH"}.issubset(arms) or not ({"AC", "AC50"} & set(arms)):
        raise ValueError("arms must include A, AH, and at least one of AC or AC50, without duplicates")
    comparisons = tuple((l, r) for l, r in ALL_COMPARISONS if l in arms and r in arms)

    errors, day_ids, leads, day_values = {}, {}, {}, {}
    all_days = set()
    ref_keys = {}
    for seed in a.seeds:
        for station in STATIONS:
            runs = {arm: _load(a.run_root / station / f"seed{seed}" / arm /
                               "validation_predictions.npz") for arm in arms}
            base = runs["A"]
            for arm in (x for x in arms if x != "A"):
                if base["keys"] != runs[arm]["keys"] or not np.array_equal(base["truth"], runs[arm]["truth"]):
                    raise ValueError(f"unpaired keys/truth for {station}/seed{seed}/{arm}")
            if any(k[0] != station for k in base["keys"]):
                raise ValueError(f"station mismatch in {station}/seed{seed}")
            if station in ref_keys and base["keys"] != ref_keys[station]:
                raise ValueError(f"validation keys differ across seeds for {station}")
            ref_keys.setdefault(station, base["keys"])
            all_days.update(base["days"].tolist())
            day_values[(seed, station)] = base["days"]
            for arm in arms:
                key = (seed, station, arm)
                errors[key] = runs[arm]["pred"] - base["truth"]
                if arm == "A":
                    id_key = (seed, station)
                    leads[id_key] = base["leads"]
    ref_days = sorted(all_days)
    if len(ref_days) < 2:
        raise ValueError("too few BJT target-date blocks")
    day_index = {day: i for i, day in enumerate(ref_days)}
    for key, values in day_values.items():
        day_ids[key] = np.asarray([day_index[d] for d in values], dtype=np.int32)
    resamples = np.random.default_rng(a.bootstrap_seed).integers(
        0, len(ref_days), size=(a.bootstrap, len(ref_days)), dtype=np.int32)
    horizons = summarize(errors, day_ids, leads, ref_days, STATIONS, a.seeds, arms, comparisons, resamples)
    report = {
        "state": "COMPLETE_DATE_BLOCK_FUTURE_GHI_HORIZON_AUDIT",
        "scope": "validation screening pilot; in-sample training forecast inputs; not paper-grade",
        "metric": "equal-station mean of seed-averaged RMSE; negative delta favors left arm",
        "arms": {arm: {"A": "no COT", "AH": "observed-history COT only",
                       "AC": "history plus full forecast-AGRI COT",
                       "AC50": "history plus 50% future normalized-COT anomaly"}[arm] for arm in arms},
        "primary_future_cot_contrast": "AC_minus_AH" if "AC" in arms else "AC50_minus_AH",
        "stations": list(STATIONS), "seeds": a.seeds,
        "target_bjt_date_blocks": len(ref_days), "bootstrap_replicates": a.bootstrap,
        "bootstrap_seed": a.bootstrap_seed, "horizons": horizons,
        "test_payload_opened": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "target_bjt_date_blocks": len(ref_days),
                      "horizons": horizons, "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
