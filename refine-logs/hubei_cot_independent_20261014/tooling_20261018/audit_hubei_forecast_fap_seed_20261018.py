#!/usr/bin/env python3
"""Independently re-score one Hubei forecast-FAP validation seed.

Only saved train/validation artifacts are read. The script verifies matching
station/time keys, truth and clear-sky values against FA/FAC, recomputes metrics
from the saved prediction arrays, and binds the receipt to file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_PACK = "9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd"
EXPECTED_TRAINER = "fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999"
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
REFERENCE_ARMS = ("FA", "FAC")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {
            "station", "init_time_bjt", "target_time_bjt", "lead_minutes",
            "truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2",
        }
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        out = {name: z[name].copy() for name in required}
    n = len(out["truth_ghi_wm2"])
    if any(len(v) != n for v in out.values()):
        raise ValueError(f"{path}: field lengths differ")
    for name in ("truth_ghi_wm2", "pred_ghi_wm2", "clear_sky_ghi_wm2"):
        if not np.isfinite(out[name]).all():
            raise ValueError(f"{path}: non-finite {name}")
    keys = list(zip(out["station"], out["init_time_bjt"], out["target_time_bjt"], out["lead_minutes"]))
    if len(set(keys)) != n:
        raise ValueError(f"{path}: duplicate forecast keys")
    out["keys"] = np.asarray(keys, dtype=object)
    return out


def key_hash(bundle: dict[str, np.ndarray]) -> str:
    canonical = "\n".join("|".join(map(str, row)) for row in bundle["keys"])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    base = a.workspace / "refine-logs/hubei_cot_independent_20261014/audit_receipts"
    fap_root = base / f"forecast_fap_seed{a.seed}_20261018" / f"forecast_FAP_seed{a.seed}_normreload_20261018"
    fa_root = base / f"forecast_ghi_normreload_seed{a.seed}_20261017" / "run"
    meta_path = fap_root / "run_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "state": "HUBEI_GHI_HEADS_SEED_COMPLETE",
        "seed": a.seed,
        "epochs": 30,
        "batch_size": 1536,
        "input_mode": "forecast",
        "arms": ["FAP"],
        "pack_sha256": EXPECTED_PACK,
        "trainer_sha256": EXPECTED_TRAINER,
        "test_payloads_read": 0,
    }
    for k, v in expected.items():
        if metadata.get(k) != v:
            raise ValueError(f"run metadata mismatch: {k}={metadata.get(k)!r}; expected {v!r}")
    if metadata.get("buffer_payloads_read", 0) != 0:
        raise ValueError("buffer_payloads_read is not zero")

    station_results: dict[str, dict[str, object]] = {}
    arm_sums: dict[str, list[float]] = {"FA": [], "FAC": [], "FAP": []}
    for station in STATIONS:
        loaded: dict[str, dict[str, np.ndarray]] = {}
        paths: dict[str, Path] = {}
        paths["FAP"] = fap_root / station / "FAP_validation_predictions_with_keys.npz"
        for arm in REFERENCE_ARMS:
            paths[arm] = fa_root / station / f"{arm}_validation_predictions_with_keys.npz"
        for arm, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded[arm] = load_bundle(path)
            if not np.all(loaded[arm]["station"] == station):
                raise ValueError(f"{station}/{arm}: station field disagrees with directory")

        ref = loaded["FA"]
        ref_keys = ref["keys"]
        row: dict[str, object] = {"n": int(len(ref_keys)), "keys_sha256": key_hash(ref)}
        for arm in ("FAC", "FAP"):
            cand = loaded[arm]
            if not np.array_equal(cand["keys"], ref_keys):
                raise ValueError(f"{station}/{arm}: key order/support differs from FA")
            for name in ("truth_ghi_wm2", "clear_sky_ghi_wm2"):
                if not np.array_equal(cand[name], ref[name]):
                    raise ValueError(f"{station}/{arm}: {name} differs from FA")

        for arm, bundle in loaded.items():
            err = bundle["pred_ghi_wm2"] - ref["truth_ghi_wm2"]
            rmse = float(np.sqrt(np.mean(np.square(err), dtype=np.float64)))
            mae = float(np.mean(np.abs(err), dtype=np.float64))
            bias = float(np.mean(err, dtype=np.float64))
            row[arm] = {"rmse_wm2": rmse, "mae_wm2": mae, "bias_wm2": bias,
                        "prediction_sha256": sha256(paths[arm])}
            arm_sums[arm].append(rmse)
        row["paired_rmse_differences_wm2"] = {
            "FAP_minus_FA": row["FAP"]["rmse_wm2"] - row["FA"]["rmse_wm2"],
            "FAP_minus_FAC": row["FAP"]["rmse_wm2"] - row["FAC"]["rmse_wm2"],
        }
        station_results[station] = row

    equal_station_rmse = {arm: float(np.mean(vals, dtype=np.float64)) for arm, vals in arm_sums.items()}
    receipt = {
        "state": "HUBEI_FORECAST_FAP_SEED_VALIDATION_INDEPENDENT_RECHECK_TEST_CLOSED",
        "scope": "single-seed validation only; not independent test or confirmatory multi-seed evidence",
        "seed": a.seed,
        "stations": list(STATIONS),
        "epochs": 30,
        "batch_size": 1536,
        "input_mode": "forecast",
        "pack_sha256": EXPECTED_PACK,
        "trainer_sha256": EXPECTED_TRAINER,
        "run_metadata_sha256": sha256(meta_path),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "equal_station_mean_rmse_wm2": equal_station_rmse,
        "equal_station_differences_wm2": {
            "FAP_minus_FA": equal_station_rmse["FAP"] - equal_station_rmse["FA"],
            "FAP_minus_FAC": equal_station_rmse["FAP"] - equal_station_rmse["FAC"],
        },
        "per_station": station_results,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "state": receipt["state"],
        "seed": a.seed,
        "equal_station_mean_rmse_wm2": equal_station_rmse,
        "equal_station_differences_wm2": receipt["equal_station_differences_wm2"],
        "output": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
