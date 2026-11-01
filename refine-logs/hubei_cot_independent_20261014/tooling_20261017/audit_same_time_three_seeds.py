#!/usr/bin/env python3
"""Independent, paired same-time validation audit for Hubei seeds 42/43/44."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "audit_receipts"
SEED_DIRS = {
    42: RECEIPTS / "same_time_seed42_independent_recheck",
    43: RECEIPTS / "same_time_seed43_44_independent_recheck" / "seed43",
    44: RECEIPTS / "same_time_seed43_44_independent_recheck" / "seed44",
}
RUN_METADATA = {
    42: SEED_DIRS[42] / "run_metadata.json",
    43: SEED_DIRS[43] / "run_metadata.json",
    44: SEED_DIRS[44] / "run_metadata.json",
}
OUT_DIR = RECEIPTS / "same_time_three_seeds_recheck"
OUT = OUT_DIR / "independent_three_seed_validation_audit.json"
SEEDS = (42, 43, 44)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
COMPARISONS = (("A", "AC"), ("A", "AP"), ("AP", "AC"))
N_BOOT = 10_000
BOOT_SEED = 20261013


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weighted_rmse(err: np.ndarray, weights: np.ndarray) -> float:
    den = float(weights.sum())
    if den <= 0:
        raise ValueError("bootstrap replicate has no supported rows")
    return float(np.sqrt(np.sum(weights * err ** 2) / den))


def weighted_mae(err: np.ndarray, weights: np.ndarray) -> float:
    den = float(weights.sum())
    if den <= 0:
        raise ValueError("bootstrap replicate has no supported rows")
    return float(np.sum(weights * np.abs(err)) / den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-root", type=Path, default=None,
                    help="optional parent with same_time_seed{42,43,44}_normreload_v1 directories")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--trainer-sha256", default=None,
                    help="require this trainer SHA for all three seeds")
    ap.add_argument("--coldstart-audit", type=Path, default=None,
                    help="bind prediction bundles to a passing cold-start/norm-reload receipt")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_dirs = dict(SEED_DIRS)
    run_metadata = dict(RUN_METADATA)
    if args.seed_root is not None:
        seed_dirs = {s: args.seed_root / f"same_time_seed{s}_normreload_v1" for s in SEEDS}
        run_metadata = {s: seed_dirs[s] / "run_metadata.json" for s in SEEDS}
    coldstart = None
    coldstart_by_key = {}
    if args.coldstart_audit is not None:
        coldstart = json.loads(args.coldstart_audit.read_text(encoding="utf-8"))
        assert coldstart["state"] == "HUBEI_CHECKPOINT_COLDSTART_NORMRELOAD_AUDIT_PASS_TEST_CLOSED"
        assert coldstart["input_mode"] == "same_time"
        assert tuple(coldstart["seeds"]) == SEEDS
        assert tuple(coldstart["arms"]) == ARMS
        assert coldstart["test_payloads_read"] == 0
        assert coldstart["buffer_payloads_read"] == 0
        assert coldstart["pack_report_sha256"] == "e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515"
        for item in coldstart["results"]:
            key = (int(item["seed"]), item["station"], item["arm"])
            assert key not in coldstart_by_key
            coldstart_by_key[key] = item
        assert len(coldstart_by_key) == len(SEEDS) * len(STATIONS) * len(ARMS)
    data: dict[tuple[int, str, str], dict] = {}
    per_seed = {}
    key_hashes = {}
    provenance = {}
    all_dates: set[str] = set()

    for seed in SEEDS:
        run_meta = json.loads(run_metadata[seed].read_text(encoding="utf-8"))
        assert run_meta["state"] == "HUBEI_GHI_HEADS_SEED_COMPLETE"
        assert run_meta["seed"] == seed and run_meta["test_payloads_read"] == 0
        assert run_meta["pack_sha256"] == "e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515"
        assert run_meta["split_sha256"] == "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
        if coldstart is not None:
            proof_run = coldstart["run_provenance"][str(seed)]
            assert proof_run["run_metadata_sha256"] == sha256(run_metadata[seed])
            assert proof_run["normalization_contract_sha256"] == run_meta["normalization_contract_sha256"]
        if args.trainer_sha256 is not None:
            assert run_meta["trainer_sha256"] == args.trainer_sha256
        provenance[str(seed)] = {
            "pack_sha256": run_meta["pack_sha256"],
            "split_sha256": run_meta["split_sha256"],
            "trainer_sha256": run_meta["trainer_sha256"],
            "normalization_contract_sha256": run_meta.get("normalization_contract_sha256"),
            "run_metadata_sha256": sha256(run_metadata[seed]),
        }
        per_seed[str(seed)] = {}
        key_hashes[str(seed)] = {}
        for station in STATIONS:
            per_seed[str(seed)][station] = {}
            seed_dir = seed_dirs[seed]
            base = seed_dir / station
            # scp places the first station's contents directly in a newly created seed dir.
            if not base.is_dir() and station == "Niushou":
                base = seed_dir
            arm_arrays = {}
            arm_files = {}
            for arm in ARMS:
                bundle = base / f"{arm}_validation_predictions_with_keys.npz"
                meta_path = base / f"{arm}_validation.json"
                z = np.load(bundle, allow_pickle=False)
                arr = {k: z[k] for k in z.files}
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                assert meta["test_payloads_read"] == 0
                assert meta["input_mode"] == "same_time"
                bundle_sha = sha256(bundle)
                checkpoint_sha = meta["checkpoint_sha256"]
                if coldstart is None:
                    assert meta.get("checkpoint_reload_parity") == "PASS"
                else:
                    proof = coldstart_by_key[(seed, station, arm)]
                    assert proof["prediction_bundle_sha256"] == bundle_sha
                    assert proof["checkpoint_sha256"] == checkpoint_sha
                    assert proof["max_abs_delta_wm2"] == 0.0
                    assert proof["cold_start_normalization_from_checkpoint"] is True
                    assert proof["n"] == meta["val_n"]
                    assert proof["validation_inference_batch_size"] == 512
                assert meta.get("normalization_contract_sha256") == run_meta.get("normalization_contract_sha256")
                assert (meta["station"], meta["arm"], meta["seed"]) == (station, arm, seed)
                assert len(arr["truth_ghi_wm2"]) == meta["val_n"]
                assert np.isfinite(arr["truth_ghi_wm2"]).all()
                assert np.isfinite(arr["pred_ghi_wm2"]).all()
                arm_arrays[arm] = arr
                arm_files[arm] = {
                    "bundle_sha256": bundle_sha,
                    "metadata_sha256": sha256(meta_path),
                    "checkpoint_sha256": meta["checkpoint_sha256"],
                }

            ref = arm_arrays["A"]
            times = ref["target_time_bjt"].astype(str)
            station_ids = ref["station"].astype(str)
            assert np.all(station_ids == station)
            keys = np.char.add(np.char.add(station_ids, "|"), times)
            assert len(np.unique(keys)) == len(keys), f"duplicate key {seed} {station}"
            dates = np.array([t[:8] for t in times])
            all_dates.update(dates.tolist())
            key_hashes[str(seed)][station] = hashlib.sha256("\n".join(keys.tolist()).encode()).hexdigest()
            truth = ref["truth_ghi_wm2"].astype(np.float64)
            clear = ref["clear_sky_ghi_wm2"].astype(np.float64)
            arm_errors = {}
            for arm in ARMS:
                arr = arm_arrays[arm]
                assert np.array_equal(arr["station"], ref["station"]), f"station/key mismatch {seed} {station} {arm}"
                assert np.array_equal(arr["target_time_bjt"], ref["target_time_bjt"]), f"time mismatch {seed} {station} {arm}"
                assert np.array_equal(arr["truth_ghi_wm2"], ref["truth_ghi_wm2"]), f"truth mismatch {seed} {station} {arm}"
                assert np.array_equal(arr["clear_sky_ghi_wm2"], ref["clear_sky_ghi_wm2"]), f"clear-sky mismatch {seed} {station} {arm}"
                err = arr["pred_ghi_wm2"].astype(np.float64) - truth
                arm_errors[arm] = err
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mae = float(np.mean(np.abs(err)))
                meta = json.loads((base / f"{arm}_validation.json").read_text(encoding="utf-8"))
                assert abs(rmse - meta["rmse_wm2"]) < 1e-8
                assert abs(mae - meta["mae_wm2"]) < 1e-8
                per_seed[str(seed)][station][arm] = {"n": int(len(err)), "rmse_wm2": rmse, "mae_wm2": mae}
            data[(seed, station, "_meta")] = {
                "times": times,
                "dates": dates,
                "keys": keys,
                "truth": truth,
                "clear": clear,
                "errors": arm_errors,
                "files": arm_files,
            }

    # All seeds must use exactly the same paired validation keys for each station.
    for station in STATIONS:
        base_meta = data[(42, station, "_meta")]
        base_keys = base_meta["keys"]
        for seed in (43, 44):
            cur_meta = data[(seed, station, "_meta")]
            assert np.array_equal(base_keys, cur_meta["keys"]), f"seed key drift {seed} {station}"
            assert np.array_equal(base_meta["truth"], cur_meta["truth"]), f"seed truth drift {seed} {station}"
            assert np.array_equal(base_meta["clear"], cur_meta["clear"]), f"seed clear-sky drift {seed} {station}"

    dates = np.array(sorted(all_dates))
    date_to_ix = {d: i for i, d in enumerate(dates)}
    groups = {}
    for seed in SEEDS:
        for station in STATIONS:
            meta = data[(seed, station, "_meta")]
            ix = np.array([date_to_ix[d] for d in meta["dates"]], dtype=np.int64)
            groups[(seed, station)] = ix

    # Point estimates follow the preregistered endpoint: equal mean across stations within seed,
    # then equal mean across three training seeds.
    def endpoint(metric: str, date_counts: np.ndarray | None = None):
        by_comparison = {}
        for arm_a, arm_b in COMPARISONS:
            station_seed_values = []
            for seed in SEEDS:
                station_values = []
                for station in STATIONS:
                    meta = data[(seed, station, "_meta")]
                    w = np.ones(len(meta["dates"]), dtype=np.float64) if date_counts is None else date_counts[groups[(seed, station)]]
                    ea = meta["errors"][arm_a]
                    eb = meta["errors"][arm_b]
                    fn = weighted_rmse if metric == "rmse" else weighted_mae
                    station_values.append(fn(eb, w) - fn(ea, w))
                station_seed_values.append(float(np.mean(station_values)))
            by_comparison[f"{arm_b}_minus_{arm_a}"] = float(np.mean(station_seed_values))
        return by_comparison

    point_rmse = endpoint("rmse")
    point_mae = endpoint("mae")
    # Run metadata is expected to differ by seed and must be retained as
    # per-seed provenance, but the scientific input/training contract must match.
    for key in ("pack_sha256", "split_sha256", "trainer_sha256", "normalization_contract_sha256"):
        if len({p[key] for p in provenance.values()}) != 1:
            raise AssertionError(f"three same-time seeds do not share one {key}")
    rng = np.random.default_rng(BOOT_SEED)
    day_draws = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    date_boot = {key: [] for key in point_rmse}
    for draw in day_draws:
        counts = np.bincount(draw, minlength=len(dates)).astype(np.float64)
        for key, value in endpoint("rmse", counts).items():
            date_boot[key].append(value)
    date_ci = {key: [float(x) for x in np.quantile(vals, [0.025, 0.975])] for key, vals in date_boot.items()}

    # Paired two-consecutive-calendar-day block sensitivity; use the same sampled blocks everywhere.
    date_set = set(dates.tolist())
    two_day_blocks = []
    from datetime import datetime, timedelta
    for d in dates:
        next_day = (datetime.strptime(d, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        if next_day in date_set:
            two_day_blocks.append((d, next_day))
    if not two_day_blocks:
        raise ValueError("validation dates contain no consecutive two-day block")
    block_boot = {key: [] for key in point_rmse}
    n_blocks = int(np.ceil(len(dates) / 2))
    for _ in range(N_BOOT):
        selected = rng.integers(0, len(two_day_blocks), size=n_blocks)
        sampled = [d for j in selected for d in two_day_blocks[j]]
        counts = np.zeros(len(dates), dtype=np.float64)
        for d in sampled:
            counts[date_to_ix[d]] += 1
        for key, value in endpoint("rmse", counts).items():
            block_boot[key].append(value)
    block_ci = {key: [float(x) for x in np.quantile(vals, [0.025, 0.975])] for key, vals in block_boot.items()}

    station_mean = {}
    for station in STATIONS:
        station_mean[station] = {}
        for arm in ARMS:
            rmses = [per_seed[str(seed)][station][arm]["rmse_wm2"] for seed in SEEDS]
            maes = [per_seed[str(seed)][station][arm]["mae_wm2"] for seed in SEEDS]
            station_mean[station][arm] = {
                "rmse_mean_wm2": float(np.mean(rmses)),
                "rmse_sd_across_seeds_wm2": float(np.std(rmses, ddof=1)),
                "mae_mean_wm2": float(np.mean(maes)),
                "mae_sd_across_seeds_wm2": float(np.std(maes, ddof=1)),
                "n_by_seed": [per_seed[str(seed)][station][arm]["n"] for seed in SEEDS],
            }

    report = {
        "state": "HUBEI_SAME_TIME_THREE_SEED_VALIDATION_AUDIT_COMPLETE_TEST_CLOSED",
        "input_mode": "same_time",
        "seeds": list(SEEDS),
        "stations": list(STATIONS),
        "arms": list(ARMS),
        "test_payloads_read": 0,
        "contract": {
            "pack_sha256": "e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515",
            "split_sha256": "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6",
            "seed_provenance": provenance,
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "metric_units": "W/m2 operating convention; instrument calibration metadata unverified",
            "metric_recomputed_from_saved_predictions": True,
            "keys_truth_and_clear_sky_identical_across_arms_and_seeds": True,
            "coldstart_audit_sha256": sha256(args.coldstart_audit) if args.coldstart_audit else None,
        },
        "primary_endpoint": "equal station mean of RMSE differences per seed, then equal mean across 3 seeds",
        "primary_comparison": "AC_minus_A; negative favors AC",
        "capacity_comparison": "AC_minus_AP; negative favors AC",
        "bootstrap": {
            "main": {"method": "paired BJT target-date cluster bootstrap; common date weights across stations, arms, and seeds", "replicates": N_BOOT, "seed": BOOT_SEED, "number_of_dates": int(len(dates)), "rmse_delta_ci95_wm2": date_ci},
            "two_day_sensitivity": {"method": "paired consecutive two-calendar-day block bootstrap", "replicates": N_BOOT, "candidate_blocks": int(len(two_day_blocks)), "rmse_delta_ci95_wm2": block_ci},
        },
        "point_estimates": {"rmse_delta_wm2": point_rmse, "mae_delta_wm2": point_mae},
        "per_seed_station_arm_metrics": per_seed,
        "station_metrics_mean_across_seeds": station_mean,
        "paired_key_hashes": key_hashes,
        "files": {str(seed): {station: data[(seed, station, "_meta")]["files"] for station in STATIONS} for seed in SEEDS},
        "run_metadata_sha256": {str(seed): sha256(run_metadata[seed]) for seed in SEEDS},
    }
    out = out_dir / "independent_three_seed_validation_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("state", "test_payloads_read", "primary_endpoint", "bootstrap", "point_estimates", "station_metrics_mean_across_seeds")}, ensure_ascii=False, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
