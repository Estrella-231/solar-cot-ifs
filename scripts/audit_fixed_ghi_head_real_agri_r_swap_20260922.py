"""Independent NumPy paired-row audit of the fixed GHI-head R swap."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def score(kt, ghi, clear):
    error = kt.astype(np.float64) * clear.astype(np.float64) - ghi.astype(np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise RuntimeError("empty/nonfinite GHI error")
    return {"rows": int(len(error)), "rmse_wm2": float(np.sqrt(np.mean(error ** 2))),
            "mae_wm2": float(np.mean(np.abs(error))), "bias_wm2": float(np.mean(error))}


def main(root, run):
    root, run = root.resolve(), run.resolve()
    complete = json.loads((run / "complete.json").read_text())
    if complete["state"] != "COMPLETE_FIXED_GHI_HEAD_REAL_AGRI_R_SWAP" or complete["test_used"] is not False:
        raise RuntimeError("fixed GHI-head swap incomplete/test-tainted")
    rows_path = root / "data/head_pack_trainval_20260908_v1/rows.csv"
    with rows_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 637902:
        raise RuntimeError("GHI head pack row count drift")
    report = {"state": "PASS_FIXED_GHI_HEAD_REAL_AGRI_R_SWAP_AUDIT", "test_used": False,
              "comparison": "same station/group/pack_row/GHI truth/clear/lead; original R FP32 vs continued R FP32",
              "stations": {}}
    for station_name in ("Sili", "Zhujia"):
        report["stations"][station_name] = {}
        for group in ("AC_H", "AC_ST"):
            saved = {}
            for scenario in ("legacy_bf16", "original_fp32", "continued_fp32"):
                with np.load(run / f"{station_name}_{group}_{scenario}_validation_predictions.npz", allow_pickle=False) as data:
                    saved[scenario] = {key: data[key] for key in
                                       ("pack_row", "pred_kt", "observed_ghi", "clear_sky_ghi", "station", "lead")}
            anchor = saved["original_fp32"]
            if len(anchor["pack_row"]) == 0 or len(np.unique(anchor["pack_row"])) != len(anchor["pack_row"]):
                raise RuntimeError("GHI validation keys empty or duplicated")
            for scenario, data in saved.items():
                for key in ("pack_row", "observed_ghi", "clear_sky_ghi", "station", "lead"):
                    if not np.array_equal(data[key], anchor[key]):
                        raise RuntimeError(f"{station_name}/{group}/{scenario} {key} mismatch")
                if not np.isfinite(data["pred_kt"]).all() or (data["pred_kt"] < 0).any():
                    raise RuntimeError("invalid predicted kt")
            indexes = anchor["pack_row"]
            labels = [rows[int(index)] for index in indexes]
            if any(label["split"] != "validation" or int(label["station_index"]) != int(station)
                   or int(label["lead_index"]) != int(lead)
                   for label, station, lead in zip(labels, anchor["station"], anchor["lead"])):
                raise RuntimeError("GHI saved row provenance mismatch")
            months = np.array([label["target_time_bjt"][:7] for label in labels])
            entry = {"rows": len(indexes), "scenarios": {}, "by_month": {}, "by_lead_minutes": {}}
            for scenario, data in saved.items():
                metrics = score(data["pred_kt"], anchor["observed_ghi"], anchor["clear_sky_ghi"])
                claimed = complete["runs"][station_name][group][scenario]["ghi_rmse"]
                if abs(metrics["rmse_wm2"] - claimed) > .001:
                    raise RuntimeError(f"{station_name}/{group}/{scenario} reported RMSE drift")
                entry["scenarios"][scenario] = metrics
            for month in sorted(set(months)):
                selected = months == month
                entry["by_month"][month] = {
                    scenario: score(data["pred_kt"][selected], anchor["observed_ghi"][selected],
                                    anchor["clear_sky_ghi"][selected])
                    for scenario, data in saved.items()}
            for lead in range(16):
                selected = anchor["lead"] == lead
                entry["by_lead_minutes"][str((lead + 1) * 15)] = {
                    scenario: score(data["pred_kt"][selected], anchor["observed_ghi"][selected],
                                    anchor["clear_sky_ghi"][selected])
                    for scenario, data in saved.items()}
            entry["continued_minus_original_fp32_rmse_wm2"] = (entry["scenarios"]["continued_fp32"]["rmse_wm2"]
                                                                - entry["scenarios"]["original_fp32"]["rmse_wm2"])
            report["stations"][station_name][group] = entry
    output = run / "independent_saved_prediction_audit.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite GHI swap audit")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"], "station_deltas": {
        station: {group: values["continued_minus_original_fp32_rmse_wm2"] for group, values in groups.items()}
        for station, groups in report["stations"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.run)
