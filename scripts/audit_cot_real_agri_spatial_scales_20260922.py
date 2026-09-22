"""Same-pixel COT error by station-centered spatial scale and validation month."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def score(predicted, reference, valid):
    error = (np.expm1(predicted.astype(np.float64)) - reference.astype(np.float64))[valid]
    if not len(error):
        return {"pixels": 0, "rmse": None}
    return {"pixels": int(len(error)), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def main(root, run):
    root, run = root.resolve(), run.resolve()
    pack = root / "data/cot_repaired_pack_20260907_v1"
    with (pack / "rows.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    ids = np.array([int(row["index"]) for row in rows])
    months = np.array([row["timestamp_bjt"][:7] for row in rows])
    paths = {"frozen_R": root / "runs/cot_repaired_seed42_v1/validation_predictions.npz",
             "continued_R": run / "log_control/validation_predictions.npz"}
    saved = {}
    for arm, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            saved[arm] = {key: data[key] for key in ("row_indices", "pred_log1p_cot", "reference_cot", "mask")}
        if not np.array_equal(saved[arm]["row_indices"], ids):
            raise RuntimeError("validation row drift")
    old, new = saved["frozen_R"], saved["continued_R"]
    for key in ("reference_cot", "mask"):
        if not np.array_equal(old[key], new[key]):
            raise RuntimeError("CPP target/mask differs")
    scopes = {"center_1x1": (8, 9), "neighborhood_3x3": (7, 10),
              "neighborhood_5x5": (6, 11), "full_16x16": (0, 16)}
    report = {"state": "COMPLETE_REAL_AGRI_COT_SPATIAL_SCALE_AUDIT", "test_used": False,
              "reference": "same real AGRI validation images, CPP COT targets and valid-pixel mask",
              "method": "RMSE over individual valid pixels in each region; no pre-averaging of predicted or CPP COT",
              "scopes": {}}
    for name, (lo, hi) in scopes.items():
        spatial = np.zeros((1, 1, 16, 16), bool)
        spatial[:, :, lo:hi, lo:hi] = True
        report["scopes"][name] = {}
        for month in ("all", *sorted(set(months))):
            selected = np.ones(len(ids), bool) if month == "all" else months == month
            target, mask = old["reference_cot"][selected], old["mask"][selected]
            strata = {"all": mask & spatial, "thick_cpp_ge30": mask & spatial & (target >= 30),
                      "clear_cpp_lt0p1": mask & spatial & (target < .1)}
            report["scopes"][name][month] = {
                label: {arm: score(values["pred_log1p_cot"][selected], target, valid)
                        for arm, values in saved.items()}
                for label, valid in strata.items()}
    output = run / "spatial_scale_audit.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite spatial scale audit")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"], "overall_scopes": {
        name: values["all"]["all"] for name, values in report["scopes"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.run)
