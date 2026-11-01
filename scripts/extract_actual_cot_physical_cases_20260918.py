"""Extract predeclared representative physical COT cases from saved validation outputs.

The two cases are chosen within the top decile of absolute AC_map prediction
adjustments: the median-magnitude improving and worsening squared-error cases.
They are diagnostic examples, never evidence of aggregate performance.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def atomic_npz(path, **value):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **value)
    temp.replace(path)


def choose(mask, squared_change):
    candidate = np.flatnonzero(mask)
    magnitude = np.abs(squared_change[candidate])
    return candidate[np.argmin(np.abs(magnitude - np.median(magnitude)))]


def locate(shards, cache, split, sequence):
    for spec in shards:
        if spec["split"] != split:
            continue
        path = cache / split / spec["name"]
        with np.load(path, allow_pickle=False) as item:
            hit = np.flatnonzero(item["indices"] == sequence)
            if len(hit):
                return {key: item[key][hit[0]].copy() for key in ("predicted_agri_physical", "cot_log1p")}
    raise RuntimeError("sequence not found in frozen cache")


def main(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise RuntimeError("refuse to overwrite case extraction")
    run = root / "runs/spatial_b1_exploratory_seed42_20260917_v1"
    base = root / "data/head_pack_trainval_20260908_v1"
    cache = root / "data/frozen_forecast_trainval_20260907_v1"
    with np.load(run / "A_sp_validation_predictions.npz", allow_pickle=False) as item:
        a = {key: item[key].copy() for key in item.files}
    with np.load(run / "AC_map_validation_predictions.npz", allow_pickle=False) as item:
        ac = {key: item[key].copy() for key in item.files}
    for key in ("pack_row", "observed_ghi", "clear_sky_ghi", "station", "lead"):
        if not np.array_equal(a[key], ac[key]):
            raise RuntimeError("paired saved prediction mismatch: " + key)
    adjustment = (ac["pred_kt"].astype(np.float64) - a["pred_kt"].astype(np.float64)) * a["clear_sky_ghi"]
    a_error = a["pred_kt"].astype(np.float64) * a["clear_sky_ghi"] - a["observed_ghi"]
    ac_error = ac["pred_kt"].astype(np.float64) * ac["clear_sky_ghi"] - ac["observed_ghi"]
    squared_change = ac_error**2 - a_error**2
    # Restrict to physically cloudier predicted cases as well as strong model
    # adjustments; this is a fixed diagnostic selection, not an aggregate test.
    c_norm = json.loads((base / "norm.json").read_text())["c"]
    c = np.load(base / "c.npy", mmap_mode="r")
    c_centre = c[a["pack_row"], 3].astype(np.float64) * c_norm["std"][3] + c_norm["mean"][3]
    strong = np.abs(adjustment) >= np.quantile(np.abs(adjustment), .90)
    cloudier = c_centre >= np.quantile(c_centre, .75)
    trajectory = a["lead"] >= 3  # retain four distinct 15-min forecast frames for the figure
    rows = {"improves": choose(strong & cloudier & trajectory & (squared_change < 0), squared_change),
            "worsens": choose(strong & cloudier & trajectory & (squared_change > 0), squared_change)}
    sequence = np.load(base / "sequence.npy", mmap_mode="r")
    station = np.load(base / "station.npy", mmap_mode="r")
    lead = np.load(base / "lead.npy", mmap_mode="r")
    split = np.load(base / "split.npy", mmap_mode="r")
    shards = json.loads((base / "source_shards.json").read_text())
    payload, metadata = {}, {}
    for outcome, local in rows.items():
        pack_row = int(a["pack_row"][local]); sequence_id, station_id, lead_id = int(sequence[pack_row]), int(station[pack_row]), int(lead[pack_row])
        if int(split[pack_row]) != 1 or station_id != int(a["station"][local]) or lead_id != int(a["lead"][local]):
            raise RuntimeError("validation row identity drift")
        source = locate(shards, cache, "val", sequence_id)
        frame_ids = np.arange(max(0, lead_id - 3), lead_id + 1, dtype=np.int64)
        if len(frame_ids) < 4:
            frame_ids = np.pad(frame_ids, (4 - len(frame_ids), 0), mode="edge")
        payload[outcome + "_agri"] = source["predicted_agri_physical"][station_id, frame_ids]
        payload[outcome + "_cot"] = source["cot_log1p"][station_id, frame_ids]
        payload[outcome + "_frame_lead"] = frame_ids
        metadata[outcome] = {"selection": "median magnitude within top-decile absolute COT-induced prediction adjustment and named squared-error sign",
                             "pack_row": pack_row, "sequence": sequence_id, "station": ["Sili", "Zhujia"][station_id],
                             "lead_minutes": int((lead_id + 1) * 15), "frame_lead_minutes": [int((item + 1) * 15) for item in frame_ids],
                             "target_predicted_COT_station_center_log1p": float(c_centre[local]),
                             "observed_GHI_Wm2": float(a["observed_ghi"][local]), "A_sp_GHI_Wm2": float(a["pred_kt"][local] * a["clear_sky_ghi"][local]),
                             "AC_map_GHI_Wm2": float(ac["pred_kt"][local] * ac["clear_sky_ghi"][local]),
                             "COT_prediction_adjustment_Wm2": float(adjustment[local]),
                             "squared_error_change_Wm4": float(squared_change[local])}
    output.mkdir(parents=True)
    atomic_npz(output / "actual_predicted_agri_cot_cases.npz", **payload)
    (output / "selection.json").write_text(json.dumps({"state": "VALIDATION_ONLY_PHYSICAL_CASES", "test_used": False,
        "source": "saved spatial B1 seed42 A_sp/AC_map predictions and frozen forecast cache", "selection": metadata}, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
