#!/usr/bin/env python3
"""Recompute descriptive month/solar-elevation strata for normreload B1 runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, required=True, help="train/validation-only pack")
    ap.add_argument("--seed-root", type=Path, required=True, help="contains same_time_seed{seed}_normreload_v1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pack_report_path = args.pack_dir / "pack_report.json"
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    if pack_report.get("test_and_buffer_values_read") != 0:
        raise ValueError("pack report does not attest test/buffer values were closed")
    rows_path = args.pack_dir / "rows.csv"
    with rows_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ghi = np.load(args.pack_dir / "ghi_wm2.npy", mmap_mode="r")
    geometry = np.load(args.pack_dir / "geometry.npy", mmap_mode="r")
    if len(rows) != len(ghi) or geometry.shape[0] != len(rows):
        raise ValueError("pack row, GHI, and geometry lengths differ")
    val_ix = [i for i, r in enumerate(rows) if r["split"] == "val"]
    if not val_ix:
        raise ValueError("empty validation set")
    key_to_ix = {}
    for i in val_ix:
        r = rows[i]
        key = (r["station"], r["time_bjt"])
        if key in key_to_ix:
            raise ValueError(f"duplicate validation pack key: {key}")
        key_to_ix[key] = i

    # Geometry channel 0 is cos(SOZ); convert to solar elevation in degrees.
    cos_soz = np.asarray(geometry[val_ix, 0, 1, 1], dtype=np.float64)
    elevation = np.degrees(np.arcsin(np.clip(cos_soz, -1.0, 1.0)))
    group_positions: dict[str, list[int]] = defaultdict(list)
    groups_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for local_i, global_i in enumerate(val_ix):
        r = rows[global_i]
        month = r["time_bjt"][:6]
        group_positions[f"month:{month}"].append(local_i)
        el = float(elevation[local_i])
        if -90 <= el < 0:
            solar_bin = "lt_0"
        elif 0 <= el < 15:
            solar_bin = "0_15"
        elif 15 <= el < 30:
            solar_bin = "15_30"
        elif 30 <= el < 45:
            solar_bin = "30_45"
        elif 45 <= el < 60:
            solar_bin = "45_60"
        elif 60 <= el <= 90.0001:
            solar_bin = "60_90"
        else:
            raise ValueError(f"solar elevation outside bins: {el}")
        group_positions[f"solar_elevation:{solar_bin}"].append(local_i)

    # Read saved keyed predictions only, and bind their truth/support back to pack rows.
    pred: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    file_hashes = {}
    for seed in SEEDS:
        seed_dir = args.seed_root / f"same_time_seed{seed}_normreload_v1"
        run_meta_path = seed_dir / "run_metadata.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta.get("test_payloads_read") != 0:
            raise ValueError(f"seed {seed} reports test payload reads")
        file_hashes[str(seed)] = {"run_metadata_sha256": sha256(run_meta_path), "stations": {}}
        for station in STATIONS:
            file_hashes[str(seed)]["stations"][station] = {}
            base = seed_dir / station
            for arm in ARMS:
                bundle = base / f"{arm}_validation_predictions_with_keys.npz"
                meta_path = base / f"{arm}_validation.json"
                z = np.load(bundle, allow_pickle=False)
                station_names = z["station"].astype(str)
                times = z["target_time_bjt"].astype(str)
                truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
                prediction = np.asarray(z["pred_ghi_wm2"], dtype=np.float64)
                if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
                    raise ValueError(f"nonfinite prediction bundle: {bundle}")
                expected_keys = [(r["station"], r["time_bjt"]) for r in rows if r["split"] == "val" and r["station"] == station]
                got_keys = list(zip(station_names.tolist(), times.tolist()))
                if got_keys != expected_keys:
                    raise ValueError(f"validation key/order mismatch: {seed} {station} {arm}")
                pack_truth = np.asarray([ghi[key_to_ix[k]] for k in expected_keys], dtype=np.float64)
                if not np.allclose(pack_truth, truth, rtol=0.0, atol=1e-4):
                    raise ValueError(f"truth does not match train/validation pack: {seed} {station} {arm}")
                if json.loads(meta_path.read_text(encoding="utf-8"))["test_payloads_read"] != 0:
                    raise ValueError(f"validation metadata reports test reads: {meta_path}")
                pred[(seed, station, arm)] = {"truth": truth, "prediction": prediction}
                file_hashes[str(seed)]["stations"][station][arm] = {
                    "bundle_sha256": sha256(bundle),
                    "metadata_sha256": sha256(meta_path),
                }

    results = {}
    for group_name, local_positions in sorted(group_positions.items()):
        chosen_global = [val_ix[j] for j in local_positions]
        selected = [(rows[i]["station"], rows[i]["time_bjt"]) for i in chosen_global]
        station_seed = []
        for station in STATIONS:
            station_keys = [k for k in selected if k[0] == station]
            for seed in SEEDS:
                arm_metrics = {}
                for arm in ARMS:
                    bundle = pred[(seed, station, arm)]
                    # Bundle order is the station's canonical validation order.
                    expected_station_keys = [(r["station"], r["time_bjt"]) for r in rows if r["split"] == "val" and r["station"] == station]
                    pos = {key: i for i, key in enumerate(expected_station_keys)}
                    sel = np.asarray([pos[k] for k in station_keys], dtype=np.int64)
                    if not len(sel):
                        continue
                    arm_metrics[arm] = rmse(bundle["truth"][sel], bundle["prediction"][sel])
                if arm_metrics:
                    station_seed.append({"station": station, "seed": seed, "n": len(station_keys), **{f"{a}_rmse_wm2": arm_metrics[a] for a in arm_metrics}})
        if len(station_seed) != len(STATIONS) * len(SEEDS):
            raise ValueError(f"not all station/seed cells represented in {group_name}")
        delta_ac_a = [x["AC_rmse_wm2"] - x["A_rmse_wm2"] for x in station_seed]
        delta_ac_ap = [x["AC_rmse_wm2"] - x["AP_rmse_wm2"] for x in station_seed]
        results[group_name] = {
            "n_by_station": {s: sum(1 for k in selected if k[0] == s) for s in STATIONS},
            "mean_station_seed_equal_AC_minus_A_rmse_wm2": float(np.mean(delta_ac_a)),
            "mean_station_seed_equal_AC_minus_AP_rmse_wm2": float(np.mean(delta_ac_ap)),
            "station_seed_metrics": station_seed,
        }

    out = {
        "state": "COMPLETE_DESCRIPTIVE_HUBEI_NORMRELOAD_VALIDATION_SUBGROUPS_TEST_CLOSED",
        "scope": "saved normreload validation predictions paired to train/validation-only pack",
        "split": "frozen Hubei monthly blocked validation; no test/buffer rows loaded",
        "test_payloads_read": 0,
        "pack_report_sha256": sha256(pack_report_path),
        "rows_sha256": sha256(rows_path),
        "geometry_sha256": sha256(args.pack_dir / "geometry.npy"),
        "ghi_sha256": sha256(args.pack_dir / "ghi_wm2.npy"),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "prediction_provenance": file_hashes,
        "solar_elevation_bins_deg": [-90, 0, 15, 30, 45, 60, 90.0001],
        "cloud_strata": "omitted: do not infer weather labels from GHI or predicted COT; independent cloud-class labels unavailable in this pack",
        "selection_warning": "best epochs were selected on validation; subgroup findings are descriptive and selection-affected",
        "metric_units": "original engineering W/m2 operating convention; instrument calibration metadata not independently verified",
        "aggregation": "RMSE paired by station/target; summarize station-seed cells equally; negative AC-control difference favors AC",
        "groups": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": out["state"], "test_payloads_read": 0, "groups": {k: {"n_by_station": v["n_by_station"], "AC_minus_A": v["mean_station_seed_equal_AC_minus_A_rmse_wm2"], "AC_minus_AP": v["mean_station_seed_equal_AC_minus_AP_rmse_wm2"]} for k, v in results.items()}, "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
