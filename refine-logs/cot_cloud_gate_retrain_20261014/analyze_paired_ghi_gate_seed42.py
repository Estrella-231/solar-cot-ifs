#!/usr/bin/env python3
"""Independently score the saved validation outputs from the paired raw/gate seed-42 run."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "paired_gate_ghi_seed42_20261015_attempt1"
OUT = RUN / "paired_date_bootstrap.json"
STATIONS = ("sili", "zhujia")
N_BOOT = 2000
RNG_SEED = 20261015


def load(station: str, arm: str) -> dict[str, np.ndarray]:
    path = RUN / station / arm / "val_keys_truth.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    e = p.astype(np.float64) - y.astype(np.float64)
    return {
        "n": int(e.size),
        "rmse_wm2": float(np.sqrt(np.mean(e * e))),
        "mae_wm2": float(np.mean(np.abs(e))),
        "bias_wm2": float(np.mean(e)),
    }


def main() -> None:
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    date_sets = []
    station_date_counts = {}
    for station in STATIONS:
        data[station] = {arm: load(station, arm) for arm in ("raw", "gate")}
        raw, gate = data[station]["raw"], data[station]["gate"]
        for key in ("station", "target_time_utc", "target_time_bjt", "ghi", "clear"):
            if not np.array_equal(raw[key], gate[key]):
                raise ValueError(f"paired key/label mismatch: {station}:{key}")
        dates = np.asarray([str(x)[:10] for x in raw["target_time_bjt"]])
        station_dates = set(dates.tolist())
        date_sets.append(station_dates)
        station_date_counts[station] = len(station_dates)
        data[station]["dates"] = dates
        np.savez_compressed(
            RUN / station / "paired_validation_predictions.npz",
            target_time_utc=raw["target_time_utc"],
            target_time_bjt=raw["target_time_bjt"],
            ghi=raw["ghi"],
            raw_prediction_ghi_wm2=raw["prediction_ghi_wm2"],
            gate_prediction_ghi_wm2=gate["prediction_ghi_wm2"],
        )
    shared_dates = sorted(set.intersection(*date_sets))
    if len(shared_dates) < 3:
        raise ValueError(f"insufficient shared target dates: {len(shared_dates)}")

    point: dict[str, dict[str, dict[str, float]]] = {}
    for station in STATIONS:
        point[station] = {}
        for arm in ("raw", "gate"):
            z = data[station][arm]
            point[station][arm] = metrics(z["ghi"], z["prediction_ghi_wm2"])
        point[station]["delta_gate_minus_raw"] = {
            key: point[station]["gate"][key] - point[station]["raw"][key]
            for key in ("rmse_wm2", "mae_wm2", "bias_wm2")
        }

    def date_rmse(station: str, arm: str, sampled: list[str]) -> float:
        z = data[station][arm]
        multiplicity = Counter(sampled)
        weights = np.asarray([multiplicity.get(d, 0) for d in data[station]["dates"]], np.float64)
        if weights.sum() <= 0:
            raise ValueError(f"bootstrap draw contains no rows for {station}")
        error = z["prediction_ghi_wm2"].astype(np.float64) - z["ghi"].astype(np.float64)
        return float(np.sqrt(np.sum(weights * error * error) / np.sum(weights)))

    rng = np.random.default_rng(RNG_SEED)
    one_day = np.empty(N_BOOT, np.float64)
    two_day = np.empty(N_BOOT, np.float64)
    station_days = {s: np.asarray(sorted(date_sets[i])) for i, s in enumerate(STATIONS)}
    consecutive = {
        s: [(i, i + 1) for i in range(len(days) - 1)
            if (np.datetime64(days[i + 1]) - np.datetime64(days[i])).astype(int) == 1]
        for s, days in station_days.items()
    }
    if any(not consecutive[s] for s in STATIONS):
        raise ValueError("no adjacent target-date pairs for two-day block sensitivity")

    for b in range(N_BOOT):
        deltas = []
        for station in STATIONS:
            sampled = rng.choice(station_days[station], size=len(station_days[station]), replace=True).tolist()
            deltas.append(date_rmse(station, "gate", sampled) - date_rmse(station, "raw", sampled))
        one_day[b] = float(np.mean(deltas))

        deltas2 = []
        for station in STATIONS:
            days = station_days[station]
            blocks = [consecutive[station][i]
                      for i in rng.integers(0, len(consecutive[station]), size=(len(days) + 1) // 2)]
            sampled2 = [str(days[j]) for pair in blocks for j in pair][:len(days)]
            deltas2.append(date_rmse(station, "gate", sampled2) - date_rmse(station, "raw", sampled2))
        two_day[b] = float(np.mean(deltas2))

    equal_station = {
        arm: float(np.mean([point[s][arm]["rmse_wm2"] for s in STATIONS]))
        for arm in ("raw", "gate")
    }
    summary = {
        "state": "INDEPENDENT_VALIDATION_RECOMPUTE_COMPLETE",
        "experiment": "paired raw COT vs conservative-gate COT, seed42",
        "validation_only": True,
        "test_payload_opened": False,
        "sample_key_and_label_match": True,
        "shared_target_dates_bjt": shared_dates,
        "shared_target_date_count": len(shared_dates),
        "station_target_date_counts": station_date_counts,
        "stations": point,
        "equal_station_mean_rmse_wm2": equal_station,
        "equal_station_delta_gate_minus_raw_wm2": equal_station["gate"] - equal_station["raw"],
        "date_cluster_bootstrap": {
            "draws": N_BOOT,
            "seed": RNG_SEED,
            "statistic": "equal-station mean of paired RMSE(gate)-RMSE(raw); within each station, resample its BJT target dates with replacement and use identical sampled dates for both arms",
            "ci95_wm2": [float(x) for x in np.quantile(one_day, [0.025, 0.975])],
            "bootstrap_mean_wm2": float(np.mean(one_day)),
        },
        "two_consecutive_date_block_sensitivity": {
            "draws": N_BOOT,
            "seed": RNG_SEED,
            "statistic": "equal-station mean of paired RMSE(gate)-RMSE(raw); within each station, resample adjacent BJT date pairs and use identical sampled dates for both arms",
            "ci95_wm2": [float(x) for x in np.quantile(two_day, [0.025, 0.975])],
            "bootstrap_mean_wm2": float(np.mean(two_day)),
            "adjacent_date_pair_count_by_station": {s: len(consecutive[s]) for s in STATIONS},
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
