#!/usr/bin/env python3
"""Validation-only, key-joined weather stratification for the matched GHI heads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def station_rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()
    repo = args.repo.resolve()
    exp = repo / "experiments/monthly_R_station16_20261013_v1"
    pack = exp / "results/ghi_head_pack_new_r_v1"
    clp_path = repo / "refine-logs/cot_value_first_20261013/clp_reference_monthly_R_seed42.npz"
    rows_path = pack / "rows.csv"
    clp = np.load(clp_path, allow_pickle=False)
    if set(clp.files) != {"clp", "clp_valid", "common_mask", "sample_indices", "station", "target_time_utc"}:
        raise RuntimeError(f"Unexpected CLP archive keys: {clp.files}")
    # The GHI head consumes the centered 3x3 field. Define weather class from
    # the co-registered CPP/CLP labels on that exact footprint, not a global mean.
    clp3 = clp["clp"][:, 0, 7:10, 7:10]
    mask3 = clp["common_mask"][:, 0, 7:10, 7:10]
    lookup = {}
    for i, (st, ts) in enumerate(zip(clp["station"].astype(str), clp["target_time_utc"].astype(str))):
        key = (st.lower(), ts)
        if key in lookup:
            raise RuntimeError(f"Duplicate CLP key {key}")
        lookup[key] = i

    with rows_path.open(encoding="utf-8", newline="") as f:
        full_rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    if len(full_rows) != 4479:
        raise RuntimeError(f"Expected 4,479 validation pack rows, found {len(full_rows)}")
    full_keys = [(r["station"].lower(), r["target_time_utc"]) for r in full_rows]
    if len(set(full_keys)) != len(full_keys):
        raise RuntimeError("Duplicate GHI validation keys")
    station_positions = {}
    running = defaultdict(int)
    for i, row in enumerate(full_rows):
        st = row["station"].lower()
        station_positions[i] = running[st]
        running[st] += 1
    # CLP exists only on its independently quality-screened common support.
    # Retain the full GHI cohort as the primary result; this report explicitly
    # scores only the keyed CLP intersection for weather-stratum diagnostics.
    keep_full = [i for i, key in enumerate(full_keys) if key in lookup]
    rows = [full_rows[i] for i in keep_full]
    keys = [full_keys[i] for i in keep_full]
    if len(rows) == 0:
        raise RuntimeError("No validation rows have keyed CPP/CLP support")
    selected_station_positions = np.asarray([station_positions[i] for i in keep_full], dtype=np.int64)
    # Require full 3x3 common support for class assignment; never fill missing
    # CLP as clear. This is a diagnostic subset, not a change to the GHI cohort.
    cloud_fraction = np.empty(len(rows), dtype=np.float64)
    for j, (r, key) in enumerate(zip(rows, keys)):
        i = lookup[key]
        valid = mask3[i]
        if valid.sum() != 9:
            cloud_fraction[j] = np.nan
        else:
            cloud_fraction[j] = np.mean(np.isin(clp3[i][valid], (1, 2)))
    strata = np.full(len(rows), "unsupported", dtype=object)
    strata[np.isfinite(cloud_fraction) & (cloud_fraction <= 0.20)] = "clear"
    strata[np.isfinite(cloud_fraction) & (cloud_fraction >= 0.80)] = "cloudy"
    strata[np.isfinite(cloud_fraction) & (cloud_fraction > 0.20) & (cloud_fraction < 0.80)] = "mixed"
    y = np.asarray([float(r["ghi"]) for r in rows], dtype=np.float64)
    station = np.asarray([r["station"].lower() for r in rows], dtype=object)
    dates = np.asarray([
        (datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat()
        for r in rows
    ], dtype=object)
    pred = {}
    for seed in (42, 43, 44):
        for st in ("sili", "zhujia"):
            d = exp / f"results/ghi_heads_new_r_seed{seed}_v1/{st}"
            for arm in ("A", "AC", "AP"):
                a = np.load(d / f"{arm}_val_prediction_ghi_wm2.npy", allow_pickle=False)
                expected_n = running[st]
                if a.shape != (expected_n,) or not np.isfinite(a).all():
                    raise RuntimeError(f"Bad predictions {seed}/{st}/{arm}: {a.shape}, expected {expected_n}")
                pred[(seed, st, arm)] = a
    # Map each keyed CLP row to its position in the full station-specific
    # prediction array (which preserves pack order), then score this subset.
    pfull = {}
    for seed in (42, 43, 44):
        for arm in ("A", "AC", "AP"):
            full = np.empty(len(rows), dtype=np.float64)
            for st in ("sili", "zhujia"):
                idx = np.flatnonzero(station == st)
                full[idx] = pred[(seed, st, arm)][selected_station_positions[idx]]
            pfull[(seed, arm)] = full

    def estimate(mask: np.ndarray, seed: int, left: str, right: str) -> float:
        rmses = []
        for st in ("sili", "zhujia"):
            use = mask & (station == st)
            if not use.any():
                return float("nan")
            rmses.append(station_rmse(y[use], pfull[(seed, left)][use]) - station_rmse(y[use], pfull[(seed, right)][use]))
        return float(np.mean(rmses))

    rng = np.random.default_rng(args.seed)
    all_dates = np.unique(dates)
    result = {
        "state": "PASS_VALIDATION_CPP_CLP_WEATHER_STRATIFICATION",
        "split": "hunan_monthly_block_2d_buffer1d_seed42_v1; validation only",
        "weather_source": "CPP CLP; clear=0, water=1, ice=2; co-produced reference, not independent truth",
        "footprint": "centered 3x3 CLP footprint matching the GHI head/COT input; no global spatial average",
        "thresholds": {"clear": "cloud fraction <= 0.20", "cloudy": "cloud fraction >= 0.80", "mixed": "0.20 < cloud fraction < 0.80"},
        "support": {},
        "cohort_relation": {
            "primary_GHI_validation_rows": len(full_rows),
            "keyed_CPP_CLP_intersection_rows": len(rows),
            "CPP_CLP_unavailable_GHI_rows": len(full_rows) - len(rows),
            "primary_GHI_cohort_unchanged": True,
        },
        "results": {},
        "bootstrap": {"replicates": args.bootstrap, "cluster": "BJT target date", "seed": args.seed},
        "provenance": {
            "pack_manifest_sha256": sha256(pack / "pack_manifest.json"),
            "rows_sha256": sha256(rows_path),
            "clp_npz_sha256": sha256(clp_path),
            "prediction_files": {},
        },
        "test_payload_opened": False,
        "caveat": "Validation-selected checkpoints; CPP CLP strata are diagnostic only and do not alter the primary common GHI cohort.",
    }
    for seed in (42, 43, 44):
        for st in ("sili", "zhujia"):
            for arm in ("A", "AC", "AP"):
                path = exp / f"results/ghi_heads_new_r_seed{seed}_v1/{st}/{arm}_val_prediction_ghi_wm2.npy"
                result["provenance"]["prediction_files"][f"seed{seed}/{st}/{arm}"] = sha256(path)
    for label in ("clear", "mixed", "cloudy", "unsupported"):
        sm = strata == label
        result["support"][label] = {
            "total": int(sm.sum()),
            "sili": int(np.sum(sm & (station == "sili"))),
            "zhujia": int(np.sum(sm & (station == "zhujia"))),
            "unique_BJT_dates": int(len(np.unique(dates[sm]))),
        }
        if label == "unsupported" or not sm.any():
            continue
        result["results"][label] = {"by_seed": {}, "three_seed_mean_delta": {}}
        for seed in (42, 43, 44):
            by_station = {}
            for st in ("sili", "zhujia"):
                use = sm & (station == st)
                by_station[st] = {
                    "n": int(use.sum()),
                    **{arm: station_rmse(y[use], pfull[(seed, arm)][use]) for arm in ("A", "AC", "AP")},
                    "AC_minus_A": station_rmse(y[use], pfull[(seed, "AC")][use]) - station_rmse(y[use], pfull[(seed, "A")][use]),
                    "AC_minus_AP": station_rmse(y[use], pfull[(seed, "AC")][use]) - station_rmse(y[use], pfull[(seed, "AP")][use]),
                } if use.any() else {"n": 0}
            deltas = {}
            for comp, left, right in (("AC_minus_A", "AC", "A"), ("AC_minus_AP", "AC", "AP")):
                seed_values = [estimate(sm, seed, left, right)]
                deltas[comp] = seed_values[0]
            result["results"][label]["by_seed"][str(seed)] = by_station
            result["results"][label].setdefault("paired_station_equal_delta_by_seed", {}).setdefault("AC_minus_A", []).append(deltas["AC_minus_A"])
            result["results"][label].setdefault("paired_station_equal_delta_by_seed", {}).setdefault("AC_minus_AP", []).append(deltas["AC_minus_AP"])
        for comp, left, right in (("AC_minus_A", "AC", "A"), ("AC_minus_AP", "AC", "AP")):
            seed_vals = result["results"][label]["paired_station_equal_delta_by_seed"][comp]
            result["results"][label]["three_seed_mean_delta"][comp] = float(np.mean(seed_vals))
            boot = []
            for _ in range(args.bootstrap):
                chosen = rng.choice(all_dates, size=len(all_dates), replace=True)
                # Keep repeated sampled dates as repeated rows; preserve paired arms/seeds/stations.
                ix = np.concatenate([np.flatnonzero(sm & (dates == day)) for day in chosen])
                if not ix.size or not all(np.any((station[ix] == st)) for st in ("sili", "zhujia")):
                    continue
                per_seed = []
                for seed in (42, 43, 44):
                    station_delta = []
                    for st in ("sili", "zhujia"):
                        sub = ix[station[ix] == st]
                        station_delta.append(station_rmse(y[sub], pfull[(seed, left)][sub]) - station_rmse(y[sub], pfull[(seed, right)][sub]))
                    per_seed.append(float(np.mean(station_delta)))
                boot.append(float(np.mean(per_seed)))
            result["results"][label].setdefault("paired_date_bootstrap_95ci", {})[comp] = [float(x) for x in np.quantile(boot, [0.025, 0.975])] if boot else None
            result["results"][label].setdefault("paired_date_bootstrap_valid_replicates", {})[comp] = len(boot)
    out = exp / "results/ghi_heads_cpp_clp_weather_validation.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "support": result["support"], "means": {k:v["three_seed_mean_delta"] for k,v in result["results"].items()}, "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
