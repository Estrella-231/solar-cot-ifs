#!/usr/bin/env python3
"""Descriptive month and target-time solar-altitude strata on validation only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "audit_receipts"
FEATURES = RECEIPTS / "same_time_subgroup_features"
SEED_DIRS = {
    42: RECEIPTS / "same_time_seed42_independent_recheck",
    43: RECEIPTS / "same_time_seed43_44_independent_recheck" / "seed43",
    44: RECEIPTS / "same_time_seed43_44_independent_recheck" / "seed44",
}
OUT = RECEIPTS / "same_time_three_seeds_recheck" / "month_solar_subgroups_validation.json"
PRIMARY_AUDIT = RECEIPTS / "same_time_three_seeds_recheck" / "independent_three_seed_validation_audit.json"
SEEDS = (42, 43, 44)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
SOLAR_EDGES = [-90, 0, 15, 30, 45, 60, 90.0001]
SOLAR_LABELS = ["below_0", "0-15", "15-30", "30-45", "45-60", "60-90"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundle_path(seed: int, station: str, arm: str) -> Path:
    base = SEED_DIRS[seed] / station
    if not base.is_dir() and station == "Niushou":
        base = SEED_DIRS[seed]
    return base / f"{arm}_validation_predictions_with_keys.npz"


def main() -> None:
    rows = pd.read_csv(FEATURES / "rows.csv", dtype={"time_bjt": str})
    val = rows.loc[rows["split"].eq("val")].copy()
    val["station"] = val["station"].astype(str)
    val["time_bjt"] = val["time_bjt"].astype(str).str.zfill(12)
    assert set(val["station"]) == set(STATIONS)
    assert val[["station", "time_bjt"]].duplicated().sum() == 0

    geom = np.load(FEATURES / "geometry.npy", mmap_mode="r")
    # The feature schema labels geometry channel 0 as cosSOZ; only validation rows are indexed/read.
    lookup = {(r.station, r.time_bjt): int(r.index) for r in val.itertuples(index=False)}
    group_by_key = {}
    for station in STATIONS:
        st = val.loc[val.station.eq(station)].copy()
        ix = np.array([lookup[(station, t)] for t in st.time_bjt], dtype=np.int64)
        cos_soz = np.asarray(geom[ix, 0, 1, 1], dtype=np.float64)
        assert np.isfinite(cos_soz).all()
        alt = np.degrees(np.arcsin(np.clip(cos_soz, -1, 1)))
        st["solar_altitude_deg"] = alt
        st["month"] = st["time_bjt"].str.slice(0, 6)
        st["solar_bin"] = pd.cut(alt, bins=SOLAR_EDGES, right=False, labels=SOLAR_LABELS)
        group_by_key[station] = st.set_index("time_bjt")

    summaries = {"month": {}, "solar_altitude_deg": {}}
    key_count = {}
    prediction_hashes = {}
    for station in STATIONS:
        all_keys = None
        per_seed_arrays = {}
        for seed in SEEDS:
            per_seed_arrays[seed] = {}
            prediction_hashes[str(seed)] = prediction_hashes.get(str(seed), {})
            prediction_hashes[str(seed)][station] = {}
            for arm in ARMS:
                pred_path = bundle_path(seed, station, arm)
                prediction_hashes[str(seed)][station][arm] = sha256(pred_path)
                z = np.load(pred_path, allow_pickle=False)
                times = z["target_time_bjt"].astype(str)
                st_ids = z["station"].astype(str)
                assert np.all(st_ids == station)
                keys = times.tolist()
                if all_keys is None:
                    all_keys = keys
                else:
                    assert keys == all_keys, f"validation support differs: {seed} {station} {arm}"
                assert all(t in group_by_key[station].index for t in keys), f"pack row missing: {seed} {station} {arm}"
                truth = z["truth_ghi_wm2"].astype(np.float64)
                pred = z["pred_ghi_wm2"].astype(np.float64)
                assert np.isfinite(truth).all() and np.isfinite(pred).all()
                per_seed_arrays[seed][arm] = pred - truth
            for arm in ARMS:
                meta_path = bundle_path(seed, station, arm).with_name(f"{arm}_validation.json")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                assert meta["test_payloads_read"] == 0

        assert all_keys is not None
        key_count[station] = len(all_keys)
        strata = group_by_key[station].loc[all_keys]
        assert int(strata["solar_bin"].notna().sum()) == len(all_keys), f"unassigned solar bin: {station}"
        assert sum(int((strata["solar_bin"].astype(str) == label).sum()) for label in SOLAR_LABELS) == len(all_keys)
        for dim in ("month", "solar_bin"):
            key_name = "month" if dim == "month" else "solar_bin"
            out_dim = summaries["month"] if dim == "month" else summaries["solar_altitude_deg"]
            values = strata[key_name].astype(str).to_numpy()
            levels = sorted(set(values))
            out_dim[station] = {}
            for level in levels:
                if level == "nan":
                    continue
                mask = values == level
                cell = {"n": int(mask.sum()), "unique_dates": int(strata.loc[mask].index.str.slice(0, 8).nunique()), "mean_solar_altitude_deg": float(strata.loc[mask, "solar_altitude_deg"].mean()), "arms_mean_over_seeds": {}, "delta_rmse_mean_over_seeds_wm2": {}}
                for arm in ARMS:
                    rmses, maes = [], []
                    for seed in SEEDS:
                        err = per_seed_arrays[seed][arm][mask]
                        rmses.append(float(np.sqrt(np.mean(err ** 2))))
                        maes.append(float(np.mean(np.abs(err))))
                    cell["arms_mean_over_seeds"][arm] = {"rmse_wm2": float(np.mean(rmses)), "mae_wm2": float(np.mean(maes))}
                for a, b in (("A", "AC"), ("AP", "AC")):
                    diffs = []
                    for seed in SEEDS:
                        ea = per_seed_arrays[seed][a][mask]
                        eb = per_seed_arrays[seed][b][mask]
                        diffs.append(float(np.sqrt(np.mean(eb ** 2)) - np.sqrt(np.mean(ea ** 2))))
                    cell["delta_rmse_mean_over_seeds_wm2"][f"{b}_minus_{a}"] = float(np.mean(diffs))
                out_dim[station][level] = cell

    primary_audit = json.loads(PRIMARY_AUDIT.read_text(encoding="utf-8"))
    assert primary_audit["state"] == "HUBEI_SAME_TIME_THREE_SEED_VALIDATION_AUDIT_COMPLETE_TEST_CLOSED"
    assert primary_audit["test_payloads_read"] == 0
    report = {
        "state": "HUBEI_SAME_TIME_MONTH_AND_SOLAR_STRATA_VALIDATION_ONLY",
        "test_payloads_read": 0,
        "seeds": list(SEEDS),
        "stations": list(STATIONS),
        "source": "same-time validation saved predictions plus target-time cosSOZ center pixel from frozen train/validation pack",
        "parent_validation_audit_sha256": sha256(PRIMARY_AUDIT),
        "frozen_pack_sha256": primary_audit["contract"]["pack_sha256"],
        "prediction_bundle_sha256": prediction_hashes,
        "geometry_channel": "cosSOZ (channel 0 of geometry.npy), solar altitude = degrees(arcsin(cosSOZ))",
        "solar_bins_deg": SOLAR_EDGES,
        "solar_bin_labels": SOLAR_LABELS,
        "solar_bin_coverage": {
            station: {
                "all_validation_prediction_keys": int(len(group_by_key[station])),
                "assigned_to_a_solar_bin": int(group_by_key[station]["solar_bin"].notna().sum()),
                "below_0_deg": int(group_by_key[station]["solar_bin"].astype(str).eq("below_0").sum()),
                "unassigned_or_out_of_range": int(group_by_key[station]["solar_bin"].isna().sum()),
            }
            for station in STATIONS
        },
        "support_by_station": key_count,
        "strata_are_descriptive_not_confirmatory": True,
        "no_cloud_condition_claim": "No independent CLP/CPP cloud-cover field was included in the GHI pack; cloud/weather strata are not inferred from GHI or prediction errors.",
        "source_hashes": {"rows_csv": sha256(FEATURES / "rows.csv"), "geometry_npy": sha256(FEATURES / "geometry.npy")},
        "strata": summaries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "support": key_count, "month": summaries["month"], "solar_altitude_deg": summaries["solar_altitude_deg"]}, ensure_ascii=False, indent=2))
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
