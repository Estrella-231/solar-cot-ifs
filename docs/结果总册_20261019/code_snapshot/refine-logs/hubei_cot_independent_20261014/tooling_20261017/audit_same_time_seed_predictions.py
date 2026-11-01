#!/usr/bin/env python3
"""Independently recompute paired validation metrics from saved keyed bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "audit_receipts" / "same_time_seed42_independent_recheck"
OUT = RUN / "independent_metric_recheck.json"
ARMS = ("A", "AC", "AP")
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
N_BOOT = 10_000
SEED = 20261017


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paired_metric_bootstrap_by_date(
    err_a: np.ndarray, err_b: np.ndarray, dates: np.ndarray, rng: np.random.Generator, metric: str
):
    unique = np.unique(dates)
    grouped_a = [err_a[dates == day] for day in unique]
    grouped_b = [err_b[dates == day] for day in unique]
    draws = rng.integers(0, len(unique), size=(N_BOOT, len(unique)))
    means = np.empty(N_BOOT, dtype=np.float64)
    for i, ix in enumerate(draws):
        sample_a = np.concatenate([grouped_a[j] for j in ix])
        sample_b = np.concatenate([grouped_b[j] for j in ix])
        if metric == "rmse":
            means[i] = np.sqrt(np.mean(sample_b ** 2)) - np.sqrt(np.mean(sample_a ** 2))
        elif metric == "mae":
            means[i] = np.mean(np.abs(sample_b)) - np.mean(np.abs(sample_a))
        else:
            raise ValueError(metric)
    return {
        "independent_date_blocks": int(len(unique)),
        "ci95_percentile": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def main() -> None:
    report = {
        "state": "HUBEI_SAME_TIME_SEED42_SAVED_PREDICTIONS_INDEPENDENT_RECHECK",
        "input_mode": "same_time",
        "seed": 42,
        "test_payloads_read": 0,
        "bootstrap": {"method": "paired target-date cluster bootstrap", "replicates": N_BOOT, "seed": SEED},
        "stations": {},
    }
    rng = np.random.default_rng(SEED)
    for station in STATIONS:
        base = RUN / station
        arrays = {}
        metas = {}
        hashes = {}
        for arm in ARMS:
            bundle = base / f"{arm}_validation_predictions_with_keys.npz"
            meta_path = base / f"{arm}_validation.json"
            z = np.load(bundle, allow_pickle=False)
            arrays[arm] = {k: z[k] for k in z.files}
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metas[arm] = meta
            hashes[arm] = {"bundle_sha256": sha256(bundle), "metadata_sha256": sha256(meta_path)}
            assert meta["test_payloads_read"] == 0
            assert meta["input_mode"] == "same_time"
            assert meta["station"] == station and meta["arm"] == arm and meta["seed"] == 42

        ref = arrays["A"]
        keys = np.char.add(np.char.add(ref["station"], "|"), ref["target_time_bjt"])
        assert len(np.unique(keys)) == len(keys), f"duplicate paired keys: {station}"
        dates = np.array([k[:8] for k in ref["target_time_bjt"]])
        truth = ref["truth_ghi_wm2"].astype(np.float64)
        clear = ref["clear_sky_ghi_wm2"].astype(np.float64)
        station_report = {
            "n": int(len(keys)),
            "unique_dates": int(len(np.unique(dates))),
            "paired_key_sha256": hashlib.sha256("\n".join(keys.tolist()).encode()).hexdigest(),
            "arms": {},
            "paired_error_differences": {},
            "bundle_hashes": hashes,
            "test_payloads_read": 0,
        }
        errors = {}
        for arm in ARMS:
            a = arrays[arm]
            for field in ("station", "target_time_bjt"):
                assert np.array_equal(ref[field], a[field]), f"key mismatch {station} {arm} {field}"
            assert np.array_equal(truth, a["truth_ghi_wm2"].astype(np.float64)), f"truth mismatch {station} {arm}"
            assert np.array_equal(clear, a["clear_sky_ghi_wm2"].astype(np.float64)), f"clear-sky mismatch {station} {arm}"
            pred = a["pred_ghi_wm2"].astype(np.float64)
            assert np.isfinite(pred).all() and np.isfinite(truth).all()
            err = pred - truth
            errors[arm] = err
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            assert abs(rmse - metas[arm]["rmse_wm2"]) < 1e-8, f"RMSE metadata mismatch {station} {arm}"
            assert abs(mae - metas[arm]["mae_wm2"]) < 1e-8, f"MAE metadata mismatch {station} {arm}"
            station_report["arms"][arm] = {"rmse_wm2": rmse, "mae_wm2": mae, "metadata_check": "PASS"}

        for arm_a, arm_b in (("A", "AC"), ("A", "AP"), ("AP", "AC")):
            station_report["paired_error_differences"][f"{arm_b}_minus_{arm_a}"] = {
                "delta_rmse_wm2": float(station_report["arms"][arm_b]["rmse_wm2"] - station_report["arms"][arm_a]["rmse_wm2"]),
                "delta_mae_wm2": float(station_report["arms"][arm_b]["mae_wm2"] - station_report["arms"][arm_a]["mae_wm2"]),
                "delta_rmse_ci95_by_date_wm2": paired_metric_bootstrap_by_date(errors[arm_a], errors[arm_b], dates, rng, "rmse"),
                "delta_mae_ci95_by_date_wm2": paired_metric_bootstrap_by_date(errors[arm_a], errors[arm_b], dates, rng, "mae"),
            }
        report["stations"][station] = station_report

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
