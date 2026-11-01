#!/usr/bin/env python3
"""Paired Hunan validation analysis for high-sun GHI and daily irradiation.

Uses only the frozen monthly-split validation pack and saved A/AC/AP predictions.
No training or test payload is accessed. Daily irradiation is scored only when
every 15-minute target interval with solar elevation > 0 is present at both
Hunan stations on the same BJT date; incomplete days are never interpolated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/monthly_R_station16_20261013_v1"
RESULTS = EXP / "results"
PACK = RESULTS / "ghi_head_pack_new_r_v1"
OUT = ROOT / "paper/analyses/solar_resource_validation_20261017"
SEEDS = (42, 43, 44)
ARMS = ("A", "AC", "AP")
STATIONS = ("sili", "zhujia")
COORDS = {
    "sili": (25.9478611111, 112.347916667),
    "zhujia": (29.566666667, 112.45),
}
N_BOOT = 2000
BOOT_SEED = 20261017


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def solar_cos_zenith_utc(ts: pd.Timestamp, lat: float, lon: float) -> float:
    """NOAA fractional-year approximation; UTC time and east-positive degrees."""
    doy = ts.dayofyear
    hour = ts.hour + ts.minute / 60 + ts.second / 3600
    gamma = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    true_solar_min = (hour * 60 + eqtime + 4 * lon) % 1440
    hour_angle = math.radians(true_solar_min / 4 - 180)
    latitude = math.radians(lat)
    return (
        math.sin(latitude) * math.sin(decl)
        + math.cos(latitude) * math.cos(decl) * math.cos(hour_angle)
    )


def expected_daylight_targets(date_bjt: str, station: str) -> pd.DatetimeIndex:
    """Canonical UTC target times on a 15-min grid with solar elevation > 0."""
    lat, lon = COORDS[station]
    start = pd.Timestamp(f"{date_bjt} 00:00:00")
    out = []
    for minute in range(0, 1440, 15):
        bjt = start + pd.Timedelta(minutes=minute)
        utc = bjt - pd.Timedelta(hours=8)
        if solar_cos_zenith_utc(utc, lat, lon) > 0:
            out.append(utc)
    return pd.DatetimeIndex(out)


def date_bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def station_equal_rmse(frame: pd.DataFrame, error_col: str) -> float:
    values = []
    for station in STATIONS:
        e = frame.loc[frame.station.eq(station), error_col].to_numpy(dtype=np.float64)
        values.append(float(np.sqrt(np.mean(e * e))))
    return float(np.mean(values))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = PACK / "rows.csv"
    geometry_path = PACK / "geometry.npy"
    rows_all = pd.read_csv(rows_path)
    val_mask = rows_all.split.eq("val").to_numpy()
    rows = rows_all.loc[val_mask].copy().reset_index(drop=True)
    geometry = np.load(geometry_path, mmap_mode="r")
    if geometry.shape[0] != len(rows_all) or geometry.shape[1:] != (2, 3, 3):
        raise RuntimeError(f"unexpected geometry shape {geometry.shape}")
    rows["target_time_utc"] = pd.to_datetime(rows.target_time_utc, utc=False)
    rows["target_time_bjt"] = rows.target_time_utc + pd.Timedelta(hours=8)
    rows["date_bjt"] = rows.target_time_bjt.dt.strftime("%Y-%m-%d")
    # Channel 0 is physical cos(SOZ); station input is spatially co-registered.
    rows["cos_soz"] = geometry[val_mask, 0, 1, 1].astype(np.float64)
    rows["solar_altitude_deg"] = np.degrees(np.arcsin(np.clip(rows.cos_soz, -1, 1)))
    common_dates = set.intersection(
        *(set(rows.loc[rows.station.eq(s), "date_bjt"].unique()) for s in STATIONS)
    )
    rows = rows[rows.date_bjt.isin(common_dates)].copy().reset_index(drop=True)

    pred: dict[tuple[int, str, str], np.ndarray] = {}
    validation_audit_path = RESULTS / "ghi_heads_three_seed_validation_audit.json"
    validation_audit = json.loads(validation_audit_path.read_text(encoding="utf-8"))
    for seed in SEEDS:
        for station in STATIONS:
            station_rows = rows.loc[rows.station.eq(station)]
            # Source arrays are station-specific, saved in the same val-pack row order.
            for arm in ARMS:
                path = RESULTS / f"ghi_heads_new_r_seed{seed}_v1/{station}/{arm}_val_prediction_ghi_wm2.npy"
                arr_all = np.load(path)
                # The pack stores station-specific validation entries contiguously in
                # each output file. Restrict by station dates in the exact CSV order.
                all_val_station = rows_all.loc[val_mask & rows_all.station.eq(station)]
                # Keep the broader station validation predictions, then key-join below.
                if len(arr_all) != len(all_val_station):
                    raise RuntimeError(f"prediction/row length mismatch: {path}")
                labels = all_val_station.ghi.to_numpy(dtype=np.float64)
                recomputed_rmse = float(np.sqrt(np.mean((arr_all - labels) ** 2)))
                reference_rmse = float(
                    validation_audit["per_station_metrics"][str(seed)][station][arm]["rmse_wm2"]
                )
                if not np.isclose(recomputed_rmse, reference_rmse, rtol=0, atol=1e-6):
                    raise RuntimeError(
                        f"prediction/key alignment witness failed for {seed}/{station}/{arm}: "
                        f"{recomputed_rmse} vs {reference_rmse}"
                    )
                pred[(seed, station, arm)] = arr_all

    # Align arrays to the restricted, common-date data by station-specific row order.
    row_indices = {}
    for station in STATIONS:
        all_station = rows_all.loc[val_mask & rows_all.station.eq(station)].copy()
        all_station["date_bjt"] = (
            pd.to_datetime(all_station.target_time_utc) + pd.Timedelta(hours=8)
        ).dt.strftime("%Y-%m-%d")
        keep = all_station.date_bjt.isin(common_dates).to_numpy()
        row_indices[station] = np.flatnonzero(keep)

    metric_rows = []
    # High-sun subgroup (solar altitude >= 45 deg), using common BJT dates.
    for seed in SEEDS:
        for station in STATIONS:
            st = rows.loc[rows.station.eq(station)].copy().reset_index(drop=True)
            ix = row_indices[station]
            high = st.solar_altitude_deg.to_numpy() >= 45.0
            for arm in ARMS:
                yhat = pred[(seed, station, arm)][ix]
                y = st.ghi.to_numpy(dtype=np.float64)
                for label, mask in (("all_common_dates", np.ones(len(st), bool)),
                                    ("solar_altitude_ge_45", high)):
                    if not mask.any():
                        continue
                    err = yhat[mask] - y[mask]
                    metric_rows.append({
                        "seed": seed, "station": station, "arm": arm,
                        "stratum": label, "n": int(mask.sum()),
                        "n_dates": int(st.loc[mask, "date_bjt"].nunique()),
                        "rmse_wm2": float(np.sqrt(np.mean(err * err))),
                        "mae_wm2": float(np.mean(np.abs(err))),
                        "bias_wm2": float(np.mean(err)),
                    })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "stratified_validation_metrics.csv", index=False)

    # Complete daylight irradiation days; use only the intersection of dates for
    # which every expected daylight label is observed at both stations.
    completeness = []
    for date in sorted(common_dates):
        okay = True
        for station in STATIONS:
            sub = rows.loc[rows.station.eq(station) & rows.date_bjt.eq(date)]
            expected = expected_daylight_targets(date, station)
            observed = pd.DatetimeIndex(sub.target_time_utc)
            is_complete = len(expected) > 0 and expected.difference(observed).empty
            completeness.append({
                "date_bjt": date, "station": station,
                "n_expected_daylight_targets": len(expected),
                "n_observed_targets": len(observed),
                "complete": bool(is_complete),
            })
            okay &= is_complete
    comp_df = pd.DataFrame(completeness)
    comp_df.to_csv(OUT / "daily_coverage_audit.csv", index=False)
    complete_dates = sorted(
        date for date, g in comp_df.groupby("date_bjt")
        if len(g) == len(STATIONS) and bool(g.complete.all())
    )

    energy_rows = []
    for date in complete_dates:
        for station in STATIONS:
            st = rows.loc[rows.station.eq(station) & rows.date_bjt.eq(date)].copy()
            expected = expected_daylight_targets(date, station)
            st = st.set_index("target_time_utc").loc[expected].reset_index()
            y_energy = float(st.ghi.sum() * 0.25 / 1000.0)
            for seed in SEEDS:
                # Array is indexed by all validation rows for the station; build
                # an exact timestamp lookup to avoid relying on DataFrame reshaping.
                all_st = rows_all.loc[val_mask & rows_all.station.eq(station)].copy().reset_index(drop=True)
                all_st["target_time_utc"] = pd.to_datetime(all_st.target_time_utc)
                key_to_i = {t: i for i, t in enumerate(all_st.target_time_utc)}
                src_i = np.array([key_to_i[t] for t in expected], dtype=int)
                for arm in ARMS:
                    yhat = pred[(seed, station, arm)][src_i]
                    pred_energy = float(yhat.sum() * 0.25 / 1000.0)
                    energy_rows.append({
                        "date_bjt": date, "station": station, "seed": seed,
                        "arm": arm, "n_daylight_targets": len(expected),
                        "observed_energy_kwh_m2": y_energy,
                        "predicted_energy_kwh_m2": pred_energy,
                        "energy_error_kwh_m2": pred_energy - y_energy,
                        "absolute_energy_error_kwh_m2": abs(pred_energy - y_energy),
                    })
    energy = pd.DataFrame(energy_rows)
    energy.to_csv(OUT / "daily_irradiation_errors.csv", index=False)

    # Aggregate day errors equally across stations, then across seeds. Bootstrap
    # dates, not 15-minute intervals or random seeds, to retain temporal clustering.
    daily = (energy.groupby(["date_bjt", "station", "arm"], as_index=False)
             .agg(abs_error_kwh_m2=("absolute_energy_error_kwh_m2", "mean"),
                  bias_kwh_m2=("energy_error_kwh_m2", "mean")))
    per_date = (daily.groupby(["date_bjt", "arm"], as_index=False)
                .agg(station_equal_abs_error=("abs_error_kwh_m2", "mean"),
                     station_equal_bias=("bias_kwh_m2", "mean")))
    energy_summary = []
    for arm in ARMS:
        x = per_date.loc[per_date.arm.eq(arm), "station_equal_abs_error"].to_numpy()
        b = per_date.loc[per_date.arm.eq(arm), "station_equal_bias"].to_numpy()
        energy_summary.append({
            "arm": arm, "n_common_complete_dates": len(x),
            "station_equal_daily_MAE_kwh_m2": float(x.mean()),
            "station_equal_daily_bias_kwh_m2": float(b.mean()),
        })
    rng = np.random.default_rng(BOOT_SEED)
    by_arm_date = per_date.pivot(index="date_bjt", columns="arm", values="station_equal_abs_error")
    e_delta = by_arm_date["AC"].to_numpy() - by_arm_date["A"].to_numpy()
    e_ci = date_bootstrap(e_delta, rng)
    # High-sun station-equal RMSE delta, with each date retained as one block.
    high_rows = []
    for seed in SEEDS:
        for station in STATIONS:
            st = rows.loc[rows.station.eq(station)].copy().reset_index(drop=True)
            ix = row_indices[station]
            mask = st.solar_altitude_deg.to_numpy() >= 45.0
            for arm in ("A", "AC"):
                yhat = pred[(seed, station, arm)][ix]
                sub = st.loc[mask, ["date_bjt", "ghi"]].copy()
                sub["sqerr"] = (yhat[mask] - sub.ghi.to_numpy()) ** 2
                high_rows.append((seed, station, arm, sub))
    date_sets = sorted(set(rows.loc[rows.solar_altitude_deg.ge(45), "date_bjt"]))
    high_delta_by_seed = []
    for seed in SEEDS:
        station_rmse = {arm: [] for arm in ("A", "AC")}
        for arm in ("A", "AC"):
            vals = []
            for _, station, a, sub in high_rows:
                if _ != seed or a != arm:
                    continue
                vals.append(float(np.sqrt(sub.sqerr.mean())))
            station_rmse[arm] = vals
        high_delta_by_seed.append(float(np.mean(station_rmse["AC"]) - np.mean(station_rmse["A"])))
    # Bootstrap on paired date-block squared-error sufficient statistics. Average
    # over seeds within a replicate; all samples on a sampled date travel together.
    rng = np.random.default_rng(BOOT_SEED)
    date_index = {d: i for i, d in enumerate(date_sets)}
    accum = {}
    for seed, station, arm, sub in high_rows:
        key = (seed, station, arm)
        sumsq = np.zeros(len(date_sets), dtype=np.float64)
        count = np.zeros(len(date_sets), dtype=np.float64)
        for date, g in sub.groupby("date_bjt"):
            i = date_index[date]
            sumsq[i] = g.sqerr.sum()
            count[i] = len(g)
        accum[key] = (sumsq, count)
    draws = rng.integers(0, len(date_sets), size=(N_BOOT, len(date_sets)))
    hd = np.empty(N_BOOT, dtype=np.float64)
    for b, dsel in enumerate(draws):
        seed_delta = []
        for seed in SEEDS:
            arm_rmses = {arm: [] for arm in ("A", "AC")}
            for station in STATIONS:
                for arm in ("A", "AC"):
                    ss, nn = accum[(seed, station, arm)]
                    den = nn[dsel].sum()
                    arm_rmses[arm].append(float(np.sqrt(ss[dsel].sum() / den)) if den else np.nan)
            seed_delta.append(float(np.nanmean(arm_rmses["AC"]) - np.nanmean(arm_rmses["A"])))
        hd[b] = float(np.mean(seed_delta))
    high_ci = tuple(float(x) for x in np.quantile(hd, [0.025, 0.975]))

    # High-sun point summaries, equal station then equal seed.
    high_summary = (metrics.query("stratum == 'solar_altitude_ge_45'")
                    .groupby("arm", as_index=False)
                    .agg(station_seed_equal_rmse_wm2=("rmse_wm2", "mean"),
                         station_seed_equal_mae_wm2=("mae_wm2", "mean"),
                         station_seed_equal_bias_wm2=("bias_wm2", "mean"),
                         sample_count=("n", "sum"), date_station_seed_count=("n_dates", "sum")))
    high_summary.to_csv(OUT / "high_sun_summary.csv", index=False)

    report = {
        "state": "PASS_VALIDATION_ONLY_PAIRED_SOLAR_RESOURCE_AUDIT",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "test_payload_opened": False,
        "training_started": False,
        "arms": list(ARMS), "seeds": list(SEEDS),
        "primary_analysis": "solar altitude >= 45 degrees, both Hunan stations, common BJT validation dates",
        "high_sun_summary": high_summary.to_dict(orient="records"),
        "high_sun_station_equal_rmse_delta_AC_minus_A_wm2_mean_seed": float(np.mean(high_delta_by_seed)),
        "high_sun_date_block_bootstrap_95ci_wm2": list(high_ci),
        "high_sun_n_common_dates": len(date_sets),
        "daily_irradiation_definition": "0.25*sum(15-minute interval-mean GHI)/1000 over all target intervals with solar elevation > 0; no interpolation",
        "daily_irradiation_n_complete_dates_both_stations": len(complete_dates),
        "daily_irradiation_summary": energy_summary,
        "daily_irradiation_delta_AC_minus_A_station_equal_mae_kwh_m2": float(e_delta.mean()) if len(e_delta) else None,
        "daily_irradiation_delta_date_block_bootstrap_95ci_kwh_m2": list(e_ci) if len(e_delta) else None,
        "daily_coverage": {s: int(comp_df.query("station == @s and complete").date_bjt.nunique()) for s in STATIONS},
        "bootstrap": {"resampling_unit": "BJT date", "replicates": N_BOOT, "seed": BOOT_SEED},
        "source_sha256": {
            "rows.csv": sha256(rows_path), "geometry.npy": sha256(geometry_path),
            "pack_manifest.json": sha256(PACK / "pack_manifest.json"),
            "three_seed_validation_audit.json": sha256(validation_audit_path),
        },
        "prediction_sha256": {
            f"seed{seed}/{station}/{arm}": sha256(RESULTS / f"ghi_heads_new_r_seed{seed}_v1/{station}/{arm}_val_prediction_ghi_wm2.npy")
            for seed in SEEDS for station in STATIONS for arm in ARMS
        },
        "notes": [
            "Regions/stations are not pooled; this audit is Hunan only.",
            "High-sun stratification is defined by the station-centre physical cos(SOZ) in the frozen GHI input pack.",
            "Daily-energy results apply only to the common complete-day subset and do not imply photovoltaic generation gain.",
            "Validation-selected checkpoints mean intervals are conditional validation uncertainty, not independent test evidence.",
        ],
    }
    (OUT / "solar_resource_validation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
