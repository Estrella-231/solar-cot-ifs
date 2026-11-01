#!/usr/bin/env python3
"""Three-seed paired validation audit for Hubei forecast FA/FAC/FAP.

This reads only saved train/validation prediction bundles and run metadata.
It does not open test/buffer data or model payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (42, 43, 44)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("FA", "FAC", "FAP")
EXPECTED_PACK = "9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd"
EXPECTED_TRAINER = "fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        fields = ("station", "init_time_bjt", "target_time_bjt", "lead_minutes",
                  "truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2")
        missing = set(fields).difference(z.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        b = {k: z[k].copy() for k in fields}
    n = len(b["truth_ghi_wm2"])
    if any(len(v) != n for v in b.values()):
        raise ValueError(f"{path}: inconsistent lengths")
    for k in ("truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2"):
        if not np.isfinite(b[k]).all():
            raise ValueError(f"{path}: non-finite {k}")
    keys = list(zip(b["init_time_bjt"].tolist(), b["target_time_bjt"].tolist(),
                    b["lead_minutes"].tolist()))
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"{path}: empty or duplicate keys")
    b["keys"] = keys  # type: ignore[assignment]
    return b


def vector_sha(values: np.ndarray) -> str:
    a = np.asarray(values, dtype="<f8")
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def key_sha(station: str, keys: list[tuple[Any, ...]]) -> str:
    text = "\n".join("|".join((station, *(str(v) for v in key))) for key in keys)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paired_day_cluster_bootstrap(
    groups: dict[tuple[int, str, str], dict[str, Any]],
    *,
    control: str,
    treatment: str,
    dates: list[str],
    resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    """Vectorized paired target-date cluster bootstrap, equal station then seed."""
    if resamples < 100:
        raise ValueError("at least 100 bootstrap resamples are required")
    date_index = {d: i for i, d in enumerate(dates)}
    per_group: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, g in groups.items():
        counts = np.zeros(len(dates), dtype=np.float64)
        sse = np.zeros(len(dates), dtype=np.float64)
        np.add.at(counts, g["date_indices"], 1.0)
        np.add.at(sse, g["date_indices"], np.square(g["error"]))
        per_group[key] = (sse, counts)

    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(dates), size=(resamples, len(dates)))
    multiplicity = np.stack([(sampled == i).sum(axis=1) for i in range(len(dates))], axis=1)
    rmse_by_arm: dict[str, dict[int, dict[str, np.ndarray]]] = {
        control: {}, treatment: {}
    }
    for arm in (control, treatment):
        for seed in (42, 43, 44):
            rmse_by_arm[arm][seed] = {}
            for station in STATIONS:
                sse, n = per_group[(seed, station, arm)]
                den = multiplicity @ n
                if np.any(den <= 0):
                    raise ValueError(f"bootstrap draw had zero support: {seed}/{station}/{arm}")
                rmse_by_arm[arm][seed][station] = np.sqrt((multiplicity @ sse) / den)

    def equal_station_seed(arm: str) -> np.ndarray:
        station_equal = [
            np.mean(np.stack([rmse_by_arm[arm][seed][st] for st in STATIONS], axis=0), axis=0)
            for seed in (42, 43, 44)
        ]
        return np.mean(np.stack(station_equal, axis=0), axis=0)

    differences = equal_station_seed(treatment) - equal_station_seed(control)
    point = float(np.mean([
        np.mean([np.sqrt(np.mean(groups[(seed, st, treatment)]["error"] ** 2)) for st in STATIONS])
        - np.mean([np.sqrt(np.mean(groups[(seed, st, control)]["error"] ** 2)) for st in STATIONS])
        for seed in (42, 43, 44)
    ]))
    return {
        "estimand": f"{treatment}-{control}; equal-station then equal-seed RMSE difference (W/m2)",
        "target_dates": dates,
        "resamples": resamples,
        "random_seed": random_seed,
        "invalid_resamples_skipped": 0,
        "estimate_difference_wm2": point,
        "bootstrap_mean_difference_wm2": float(np.mean(differences)),
        "ci95_wm2": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--random-seed", type=int, default=20261018)
    args = ap.parse_args()
    root = args.workspace / "refine-logs/hubei_cot_independent_20261014/audit_receipts"

    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    rmse: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    supports: dict[str, dict[str, Any]] = {}
    metadata_hashes: dict[str, str] = {}

    for seed in SEEDS:
        if seed in (42, 43):
            fap_root = root / f"forecast_fap_seed{seed}_20261018" / f"forecast_FAP_seed{seed}_normreload_20261018"
        else:
            fap_root = root / f"forecast_fap_seed{seed}_20261018" / f"forecast_FAP_seed{seed}_normreload_20261018"
        fa_root = root / f"forecast_ghi_normreload_seed{seed}_20261017" / "run"
        fap_meta_path = fap_root / "run_metadata.json"
        fap_meta = json.loads(fap_meta_path.read_text(encoding="utf-8"))
        fa_meta_path = fa_root / "run_metadata.json"
        fa_meta = json.loads(fa_meta_path.read_text(encoding="utf-8"))
        fa_required = {"state": "HUBEI_GHI_HEADS_SEED_COMPLETE", "seed": seed,
                       "epochs": 30, "batch_size": 1536, "input_mode": "forecast",
                       "arms": ["FA", "FAC"], "pack_sha256": EXPECTED_PACK,
                       "trainer_sha256": EXPECTED_TRAINER, "test_payloads_read": 0}
        for k, v in fa_required.items():
            if fa_meta.get(k) != v:
                raise ValueError(f"FA/FAC seed{seed} metadata mismatch {k}: {fa_meta.get(k)!r}")
        if fa_meta.get("buffer_payloads_read", 0) != 0:
            raise ValueError(f"FA/FAC seed{seed} buffer counter is nonzero")
        metadata_hashes[f"seed{seed}_FA_FAC_run_metadata"] = sha256(fa_meta_path)
        required = {"state": "HUBEI_GHI_HEADS_SEED_COMPLETE", "seed": seed,
                    "epochs": 30, "batch_size": 1536, "input_mode": "forecast",
                    "arms": ["FAP"], "pack_sha256": EXPECTED_PACK,
                    "trainer_sha256": EXPECTED_TRAINER, "test_payloads_read": 0}
        for k, v in required.items():
            if fap_meta.get(k) != v:
                raise ValueError(f"FAP seed{seed} metadata mismatch {k}: {fap_meta.get(k)!r}")
        if fap_meta.get("buffer_payloads_read", 0) != 0:
            raise ValueError(f"FAP seed{seed} buffer counter is nonzero")
        metadata_hashes[f"seed{seed}_FAP_run_metadata"] = sha256(fap_meta_path)

        for station in STATIONS:
            station_bundle: dict[str, dict[str, np.ndarray]] = {}
            path_for: dict[str, Path] = {
                "FA": fa_root / station / "FA_validation_predictions_with_keys.npz",
                "FAC": fa_root / station / "FAC_validation_predictions_with_keys.npz",
                "FAP": fap_root / station / "FAP_validation_predictions_with_keys.npz",
            }
            for arm, path in path_for.items():
                if not path.is_file():
                    raise FileNotFoundError(path)
                b = load_bundle(path)
                if not np.all(b["station"] == station):
                    raise ValueError(f"seed{seed}/{station}/{arm}: wrong station labels")
                station_bundle[arm] = b
                input_hashes[f"seed{seed}/{station}/{arm}"] = sha256(path)

            ref = station_bundle["FA"]
            ref_keys = ref["keys"]
            for arm in ("FAC", "FAP"):
                cand = station_bundle[arm]
                if cand["keys"] != ref_keys:
                    raise ValueError(f"seed{seed}/{station}/{arm}: keys differ from FA")
                for field in ("truth_ghi_wm2", "clear_sky_ghi_wm2"):
                    if not np.array_equal(cand[field], ref[field]):
                        raise ValueError(f"seed{seed}/{station}/{arm}: {field} differs")
            support_key = f"seed{seed}/{station}"
            supports[support_key] = {
                "n": len(ref_keys),
                "keys_sha256": key_sha(station, ref_keys),
                "truth_sha256": vector_sha(ref["truth_ghi_wm2"]),
                "clear_sky_sha256": vector_sha(ref["clear_sky_ghi_wm2"]),
            }
            for arm in ARMS:
                b = station_bundle[arm]
                err = b["pred_ghi_wm2"] - ref["truth_ghi_wm2"]
                m = {
                    "rmse_wm2": float(np.sqrt(np.mean(err * err, dtype=np.float64))),
                    "mae_wm2": float(np.mean(np.abs(err), dtype=np.float64)),
                    "bias_wm2": float(np.mean(err, dtype=np.float64)),
                }
                rmse[arm].setdefault(station, {})[str(seed)] = m["rmse_wm2"]
                records[(seed, station, arm)] = {
                    "keys": ref_keys,
                    "truth_wm2": ref["truth_ghi_wm2"],
                    "prediction_wm2": b["pred_ghi_wm2"],
                    "error": b["pred_ghi_wm2"] - ref["truth_ghi_wm2"],
                    "date_indices": [],
                }
                records[(seed, station, arm)]["target_dates"] = [str(k[1])[:8] for k in ref_keys]
                # Keep per station/seed metrics in the receipt structure.
                supports[support_key].setdefault("metrics", {})[arm] = m

    # Assert exact support and truth agreement across seeds as well as arms.
    for station in STATIONS:
        ref = supports[f"seed{SEEDS[0]}/{station}"]
        for seed in SEEDS[1:]:
            item = supports[f"seed{seed}/{station}"]
            for field in ("n", "keys_sha256", "truth_sha256", "clear_sky_sha256"):
                if item[field] != ref[field]:
                    raise ValueError(f"seed support mismatch {station}/{field}")

    station_seed_mean_rmse = {
        arm: {str(seed): float(np.mean([rmse[arm][s][str(seed)] for s in STATIONS], dtype=np.float64))
              for seed in SEEDS}
        for arm in ARMS
    }
    equal_station_seed_rmse = {
        arm: float(np.mean(list(station_seed_mean_rmse[arm].values()), dtype=np.float64))
        for arm in ARMS
    }
    dates = sorted({d for g in records.values() for d in g["target_dates"]})
    date_to_index = {d: i for i, d in enumerate(dates)}
    for g in records.values():
        g["date_indices"] = np.asarray([date_to_index[d] for d in g["target_dates"]], dtype=np.int64)
    boot = {
        "FAC_minus_FA": paired_day_cluster_bootstrap(
            records, control="FA", treatment="FAC", dates=dates,
            resamples=args.resamples, random_seed=args.random_seed - 1),
        "FAP_minus_FA": paired_day_cluster_bootstrap(
            records, control="FA", treatment="FAP", dates=dates,
            resamples=args.resamples, random_seed=args.random_seed),
        "FAP_minus_FAC": paired_day_cluster_bootstrap(
            records, control="FAC", treatment="FAP", dates=dates,
            resamples=args.resamples, random_seed=args.random_seed + 1),
    }
    receipt = {
        "state": "HUBEI_FORECAST_FAP_THREE_SEED_VALIDATION_RECHECK_TEST_CLOSED",
        "scope": "validation only; not independent test or confirmatory test evidence",
        "seeds": list(SEEDS), "stations": list(STATIONS), "arms": list(ARMS),
        "epochs": 30, "batch_size": 1536, "input_mode": "forecast",
        "pack_sha256": EXPECTED_PACK, "trainer_sha256": EXPECTED_TRAINER,
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "equal_station_then_seed_rmse_wm2": equal_station_seed_rmse,
        "per_seed_equal_station_rmse_wm2": station_seed_mean_rmse,
        "paired_date_cluster_bootstrap": boot,
        "supports_by_seed_station": supports,
        "input_bundle_sha256": input_hashes,
        "run_metadata_sha256": metadata_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "state": receipt["state"],
        "equal_station_then_seed_rmse_wm2": equal_station_seed_rmse,
        "per_seed_equal_station_rmse_wm2": station_seed_mean_rmse,
        "paired_date_cluster_bootstrap": boot,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
