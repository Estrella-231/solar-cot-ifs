#!/usr/bin/env python3
"""Independent FP64 audit of the saved validation predictions for attempt2.

This deliberately reads only the frozen train/validation pack's validation
rows and the 18 saved validation prediction bundles. It never opens test or
buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)
ARMS = ("FA_HIST", "FAC_HIST")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e, dtype=np.float64)))


def row_keys(z: dict[str, np.ndarray]) -> list[tuple[str, str, str, int]]:
    return list(zip(z["station"].astype(str).tolist(),
                    z["init_time_bjt"].astype(str).tolist(),
                    z["target_time_bjt"].astype(str).tolist(),
                    z["lead_minutes"].astype(np.int64).tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()

    scored = json.loads((a.scored / "score_receipt.json").read_text())
    pack_report = json.loads((a.pack / "pack_report.json").read_text())
    if scored.get("validation_only") is not True:
        raise ValueError("original score receipt is not marked validation-only")
    if scored.get("test_payloads_read") != 0 or scored.get("buffer_payloads_read") != 0:
        raise ValueError("score receipt reports test/buffer reads")
    if pack_report.get("test_payloads_read") != 0 or pack_report.get("buffer_payloads_read") != 0:
        raise ValueError("pack report reports test/buffer reads")

    with (a.pack / "rows.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]
    if len(val_rows) != 15275:
        raise ValueError(f"unexpected validation row count: {len(val_rows)}")

    expected: dict[str, set[tuple[str, str, str, int]]] = {}
    for st in STATIONS:
        expected[st] = {(r["station"], r["init_time_bjt"], r["target_time_bjt"], int(r["lead_minutes"]))
                        for r in val_rows if r["station"] == st}

    predictions: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    checks: list[dict[str, object]] = []
    with (a.scored / "arm_metrics.csv").open(newline="", encoding="utf-8") as f:
        scored_arms = list(csv.DictReader(f))
    all_hashes = {x["station"] + "/" + x["seed"] + "/" + x["arm"]: x["prediction_sha256"]
                  for x in scored_arms}
    for st in STATIONS:
        for seed in SEEDS:
            arm_bundles = {}
            paths = {}
            for arm in ARMS:
                p = a.experiment_root / "training_attempt2" / "pairs" / f"{st}_seed{seed}" / f"{arm}_validation_predictions.npz"
                actual_hash = sha256(p)
                expected_hash = all_hashes[f"{st}/{seed}/{arm}"]
                if actual_hash != expected_hash:
                    raise ValueError(f"prediction hash mismatch: {st}/{seed}/{arm}")
                with np.load(p, allow_pickle=False) as zf:
                    z = {k: np.asarray(zf[k]).copy() for k in zf.files}
                keys = row_keys(z)
                if len(keys) != len(set(keys)) or set(keys) != expected[st]:
                    raise ValueError(f"validation key set mismatch: {st}/{seed}/{arm}")
                if z["split"].astype(str).tolist() != ["val"] * len(keys):
                    raise ValueError(f"non-validation row in bundle: {st}/{seed}/{arm}")
                if not np.isfinite(z["pred_ghi_wm2"]).all() or not np.isfinite(z["truth_ghi_wm2"]).all():
                    raise ValueError(f"non-finite values: {st}/{seed}/{arm}")
                arm_bundles[arm] = (z, keys)
                paths[arm] = str(p)
            za, ka = arm_bundles[ARMS[0]]
            zc, kc = arm_bundles[ARMS[1]]
            if ka != kc:
                raise ValueError(f"paired order mismatch: {st}/{seed}")
            if not np.array_equal(za["truth_ghi_wm2"], zc["truth_ghi_wm2"]):
                raise ValueError(f"paired truth mismatch: {st}/{seed}")
            if not np.array_equal(za["clear_sky_wm2"], zc["clear_sky_wm2"]):
                raise ValueError(f"paired clear-sky mismatch: {st}/{seed}")
            predictions[(st, seed, ARMS[0])] = za
            predictions[(st, seed, ARMS[1])] = zc
            checks.append({"station": st, "seed": seed, "n_each_arm": len(ka),
                           "same_key_order": True, "same_truth": True,
                           "same_clear_sky": True, "prediction_hashes_match_score_receipt": True})

    per_pair = []
    for st in STATIONS:
        for seed in SEEDS:
            arms = {}
            for arm in ARMS:
                z = predictions[(st, seed, arm)]
                arms[arm] = rmse(z["pred_ghi_wm2"], z["truth_ghi_wm2"])
            per_pair.append({"station": st, "seed": seed,
                             "FA_HIST_rmse_wm2": arms["FA_HIST"],
                             "FAC_HIST_rmse_wm2": arms["FAC_HIST"],
                             "delta_fac_minus_fa_wm2": arms["FAC_HIST"] - arms["FA_HIST"]})

    mean_rmse = {arm: float(np.mean([p[f"{arm}_rmse_wm2"] for p in per_pair])) for arm in ARMS}
    point_delta = float(np.mean([p["delta_fac_minus_fa_wm2"] for p in per_pair]))
    dates = sorted({str(t)[:8] for z in predictions.values() for t in z["target_time_bjt"]})
    stat_index = {s: i for i, s in enumerate(STATIONS)}
    seed_index = {s: i for i, s in enumerate(SEEDS)}
    sse = np.zeros((2, len(STATIONS), len(SEEDS), len(dates)), dtype=np.float64)
    count = np.zeros_like(sse, dtype=np.int64)
    for ai, arm in enumerate(ARMS):
        for st in STATIONS:
            for seed in SEEDS:
                z = predictions[(st, seed, arm)]
                e = np.asarray(z["pred_ghi_wm2"], dtype=np.float64) - np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
                day = np.asarray([str(t)[:8] for t in z["target_time_bjt"]])
                for di, d in enumerate(dates):
                    m = day == d
                    if np.any(m):
                        sse[ai, stat_index[st], seed_index[seed], di] = np.sum(e[m] * e[m], dtype=np.float64)
                        count[ai, stat_index[st], seed_index[seed], di] = int(m.sum())

    rng = np.random.default_rng(a.seed)
    boot_delta = np.empty(a.draws, dtype=np.float64)
    boot_gain = np.empty(a.draws, dtype=np.float64)
    for b in range(a.draws):
        draw = rng.integers(0, len(dates), size=len(dates))
        n = count[..., draw].sum(axis=-1)
        ss = sse[..., draw].sum(axis=-1)
        score = np.sqrt(ss / np.maximum(n, 1))
        eq = score.mean(axis=2).mean(axis=1)
        boot_delta[b] = eq[1] - eq[0]
        boot_gain[b] = 100.0 * (eq[0] - eq[1]) / eq[0]

    old = scored["global_summary"]
    result = {
        "state": "INDEPENDENT_SAVED_VALIDATION_PREDICTION_AUDIT_PASS",
        "method": "fresh_numpy_fp64_recalculation_from_18_saved_validation_npz_bundles",
        "all_3_stations_x_3_seeds_x_2_arms_audited": True,
        "validation_rows_in_pack": len(val_rows),
        "row_counts_per_station": {s: len(expected[s]) for s in STATIONS},
        "common_key_order_truth_clear_sky_hash_checks": checks,
        "equal_station_equal_seed_mean_rmse_wm2": mean_rmse,
        "paired_delta_fac_minus_fa_wm2": point_delta,
        "relative_rmse_improvement_pct": 100.0 * (mean_rmse["FA_HIST"] - mean_rmse["FAC_HIST"]) / mean_rmse["FA_HIST"],
        "independent_date_bootstrap_95pct_delta_ci_wm2": np.quantile(boot_delta, [0.025, 0.975]).tolist(),
        "independent_date_bootstrap_95pct_improvement_ci_pct": np.quantile(boot_gain, [0.025, 0.975]).tolist(),
        "bootstrap_draws": a.draws,
        "bootstrap_dates_bjt": dates,
        "agrees_with_primary_score": {
            "FA_HIST_rmse": abs(mean_rmse["FA_HIST"] - old["equal_station_equal_seed_mean_rmse_wm2"]["FA_HIST"]) < 1e-10,
            "FAC_HIST_rmse": abs(mean_rmse["FAC_HIST"] - old["equal_station_equal_seed_mean_rmse_wm2"]["FAC_HIST"]) < 1e-10,
            "paired_delta": abs(point_delta - old["paired_delta_rmse_fac_minus_fa_wm2"]) < 1e-10,
            "delta_ci": np.allclose(np.quantile(boot_delta, [0.025, 0.975]), old["paired_bjt_date_block_bootstrap_95pct_delta_ci_wm2"], atol=1e-10, rtol=0),
            "improvement_ci": np.allclose(np.quantile(boot_gain, [0.025, 0.975]), old["paired_bjt_date_block_bootstrap_95pct_improvement_ci_pct"], atol=1e-10, rtol=0),
        },
        "validation_only": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "source_prediction_count": len(checks) * 2,
    }
    if not all(result["agrees_with_primary_score"].values()):
        raise ValueError("independent FP64 recalculation disagrees with primary score")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "state", "equal_station_equal_seed_mean_rmse_wm2", "paired_delta_fac_minus_fa_wm2",
        "relative_rmse_improvement_pct", "independent_date_bootstrap_95pct_delta_ci_wm2",
        "independent_date_bootstrap_95pct_improvement_ci_pct", "agrees_with_primary_score",
        "test_payloads_read", "buffer_payloads_read")}, indent=2))


if __name__ == "__main__":
    main()
