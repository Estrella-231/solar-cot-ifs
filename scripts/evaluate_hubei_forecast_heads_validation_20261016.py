#!/usr/bin/env python3
"""Paired validation-only evaluation for the Hubei FA/FAC forecast heads.

Expected run layout: <run-root>/seed{42,43,44}/{station}/
  {FA,FAC}_validation_predictions_with_keys.npz
The evaluator refuses missing or mismatched keys and never reads test/buffer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)
ARMS = ("FA", "FAC")
KEY_FIELDS = ("init_time_bjt", "target_time_bjt", "lead_minutes")


def load_prediction(path: Path, station: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"station", "truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2", *KEY_FIELDS}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        out = {key: np.asarray(z[key]) for key in required}
    n = len(out["truth_ghi_wm2"])
    if n == 0 or any(len(v) != n for v in out.values()):
        raise ValueError(f"Empty or inconsistent prediction bundle: {path}")
    if not np.all(out["station"] == station):
        raise ValueError(f"Station mismatch in {path}")
    if not all(np.isfinite(out[k]).all() for k in ("truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2")):
        raise ValueError(f"Nonfinite values in {path}")
    if (out["truth_ghi_wm2"] < 0).any() or (out["clear_sky_ghi_wm2"] <= 0).any():
        raise ValueError(f"Invalid validation support in {path}")
    lead = out["lead_minutes"].astype(np.int64)
    if not np.isin(lead, np.arange(15, 241, 15)).all():
        raise ValueError(f"Noncanonical forecast lead in {path}")
    keys = list(zip(out["init_time_bjt"].astype(str), out["target_time_bjt"].astype(str), lead.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate validation key in {path}")
    out["lead_minutes"] = lead
    out["keys"] = np.asarray(keys, dtype=object)
    return out


def metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    e = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if weights is None:
        return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))
    w = np.asarray(weights, dtype=np.float64)
    den = float(w.sum())
    if den <= 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.sum(w * e * e) / den)), float(np.sum(w * np.abs(e)) / den)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261016)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.bootstrap_reps < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    bundles: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    paired_diffs: list[dict[str, object]] = []
    for seed in SEEDS:
        for station in STATIONS:
            fa = load_prediction(a.run_root / f"seed{seed}" / station / "FA_validation_predictions_with_keys.npz", station)
            fac = load_prediction(a.run_root / f"seed{seed}" / station / "FAC_validation_predictions_with_keys.npz", station)
            if not np.array_equal(fa["keys"], fac["keys"]):
                raise ValueError(f"FA/FAC validation keys differ: seed={seed}, station={station}")
            for field in ("truth_ghi_wm2", "clear_sky_ghi_wm2"):
                if not np.array_equal(fa[field], fac[field]):
                    raise ValueError(f"FA/FAC {field} differs: seed={seed}, station={station}")
            bundles[(seed, station, "FA")] = fa
            bundles[(seed, station, "FAC")] = fac
            for arm, obj in (("FA", fa), ("FAC", fac)):
                rmse, mae = metrics(obj["truth_ghi_wm2"], obj["pred_ghi_wm2"])
                paired_diffs.append({"seed": seed, "station": station, "arm": arm,
                                     "n": int(len(obj["truth_ghi_wm2"])), "rmse_wm2": rmse, "mae_wm2": mae})

    # Every seed and arm must evaluate the same validation keys per station.
    for station in STATIONS:
        ref = bundles[(SEEDS[0], station, "FA")]["keys"]
        for seed in SEEDS:
            for arm in ARMS:
                if not np.array_equal(ref, bundles[(seed, station, arm)]["keys"]):
                    raise ValueError(f"Validation support differs across seeds/arms: {station}/{seed}/{arm}")

    # Primary descriptive endpoint: station-equal mean RMSE, then seed-equal mean.
    primary = {}
    station_lead = {}
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            site_rmse = []
            for station in STATIONS:
                obj = bundles[(seed, station, arm)]
                site_rmse.append(metrics(obj["truth_ghi_wm2"], obj["pred_ghi_wm2"])[0])
            per_seed.append({"seed": seed, "station_equal_rmse_wm2": float(np.mean(site_rmse))})
        primary[arm] = {"by_seed": per_seed,
                        "mean_station_equal_rmse_wm2": float(np.mean([x["station_equal_rmse_wm2"] for x in per_seed]))}
        for station in STATIONS:
            for lead in range(15, 241, 15):
                seed_scores = []
                for seed in SEEDS:
                    obj = bundles[(seed, station, arm)]
                    mask = obj["lead_minutes"] == lead
                    if mask.any():
                        r, m = metrics(obj["truth_ghi_wm2"][mask], obj["pred_ghi_wm2"][mask])
                        seed_scores.append((int(mask.sum()), r, m))
                station_lead[f"{arm}/{station}/{lead}"] = {
                    "n_per_seed": [v[0] for v in seed_scores],
                    "rmse_wm2_mean_over_seeds": float(np.mean([v[1] for v in seed_scores])) if seed_scores else None,
                    "mae_wm2_mean_over_seeds": float(np.mean([v[2] for v in seed_scores])) if seed_scores else None,
                }

    # Shared target-date bootstrap: one date weight is reused across stations,
    # model arms, leads, and seeds to preserve pairing and regional clustering.
    date_keys = sorted({str(k[1])[:8] for obj in bundles.values() for k in obj["keys"]})
    if len(date_keys) < 2:
        raise ValueError("Insufficient distinct target dates for date-block bootstrap")
    date_pos = {d: i for i, d in enumerate(date_keys)}
    rng = np.random.default_rng(a.bootstrap_seed)
    diffs_rmse, diffs_mae = [], []
    for _ in range(a.bootstrap_reps):
        draws = rng.integers(0, len(date_keys), size=len(date_keys))
        multiplicity = np.bincount(draws, minlength=len(date_keys))
        scores = {arm: {"r": [], "m": []} for arm in ARMS}
        for seed in SEEDS:
            for station in STATIONS:
                fa, fac = (bundles[(seed, station, arm)] for arm in ARMS)
                dates = np.asarray([str(k[1])[:8] for k in fa["keys"]])
                w = np.asarray([multiplicity[date_pos[d]] for d in dates], dtype=np.float64)
                for arm, obj in (("FA", fa), ("FAC", fac)):
                    r, m = metrics(obj["truth_ghi_wm2"], obj["pred_ghi_wm2"], w)
                    if np.isfinite(r) and np.isfinite(m):
                        scores[arm]["r"].append(r); scores[arm]["m"].append(m)
        rfa = float(np.mean(scores["FA"]["r"]))
        rfac = float(np.mean(scores["FAC"]["r"]))
        mfa = float(np.mean(scores["FA"]["m"]))
        mfac = float(np.mean(scores["FAC"]["m"]))
        diffs_rmse.append(rfac - rfa); diffs_mae.append(mfac - mfa)

    report = {
        "state": "HUBEI_FORECAST_HEAD_VALIDATION_PAIRED_EVAL_ONLY",
        "interpretation": "validation-only, best-epoch-selected; not independent test evidence",
        "seeds": list(SEEDS), "stations": list(STATIONS), "arms": list(ARMS),
        "primary_endpoint": "mean station RMSE within seed, then mean across seeds",
        "primary": primary,
        "FAC_minus_FA_primary_rmse_wm2": primary["FAC"]["mean_station_equal_rmse_wm2"] - primary["FA"]["mean_station_equal_rmse_wm2"],
        "per_seed_station_arm_metrics": paired_diffs,
        "per_station_lead": station_lead,
        "date_block_bootstrap": {
            "resamples": a.bootstrap_reps, "seed": a.bootstrap_seed, "unique_target_dates": len(date_keys),
            "shared_weights_across_stations_arms_leads_seeds": True,
            "FAC_minus_FA_rmse_wm2_mean": float(np.mean(diffs_rmse)),
            "FAC_minus_FA_rmse_wm2_ci95": [float(np.quantile(diffs_rmse, 0.025)), float(np.quantile(diffs_rmse, 0.975))],
            "FAC_minus_FA_mae_wm2_mean": float(np.mean(diffs_mae)),
            "FAC_minus_FA_mae_wm2_ci95": [float(np.quantile(diffs_mae, 0.025)), float(np.quantile(diffs_mae, 0.975))],
        },
        "test_and_buffer_payloads_read": 0,
        "input_files": {f"seed{s}/{st}/{arm}": str(a.run_root / f"seed{s}" / st / f"{arm}_validation_predictions_with_keys.npz")
                        for s in SEEDS for st in STATIONS for arm in ARMS},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "primary": primary,
                      "FAC_minus_FA_primary_rmse_wm2": report["FAC_minus_FA_primary_rmse_wm2"],
                      "bootstrap_ci95": report["date_block_bootstrap"]["FAC_minus_FA_rmse_wm2_ci95"],
                      "test_and_buffer_payloads_read": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
