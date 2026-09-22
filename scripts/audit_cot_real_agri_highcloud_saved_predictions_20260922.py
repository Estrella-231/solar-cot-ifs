"""Independently audit real-AGRI COT retrieval validation predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def score(prediction, reference, mask):
    error = (np.expm1(prediction.astype(np.float64)) - reference.astype(np.float64))[mask]
    if not len(error) or not np.isfinite(error).all():
        raise RuntimeError("empty/nonfinite validation stratum")
    return {"pixels": int(len(error)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}


def main(root, run):
    root, run = root.resolve(), run.resolve()
    complete = json.loads((run / "complete.json").read_text())
    contract = json.loads((run / "contract.json").read_text())
    if complete["state"] != "COMPLETE_PAIRED_VALIDATION_PILOT" or complete["test_used"] is not False:
        raise RuntimeError("real-AGRI pilot incomplete or test-tainted")
    if contract["input_domain"] != "observed real AGRI for both train and validation; no forecast AGRI":
        raise RuntimeError("not the real-AGRI experiment contract")
    pack = root / "data/cot_repaired_pack_20260907_v1"
    with (pack / "rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    val_rows = np.array([int(row["index"]) for row in rows if row["split"] == "validation"], np.int64)
    stations = np.array([rows[int(index)]["station_id"] for index in val_rows])
    if len(val_rows) != 6536 or len(set(val_rows.tolist())) != len(val_rows):
        raise RuntimeError("validation keys drifted")
    target = np.asarray(np.load(pack / "target.npy", mmap_mode="r")[val_rows], np.float32) * np.float32(100)
    mask = np.asarray(np.load(pack / "mask.npy", mmap_mode="r")[val_rows], bool)
    center = np.zeros((1, 1, 16, 16), bool)
    center[:, :, 6:11, 6:11] = True
    report = {"state": "PASS_REAL_AGRI_COT_SAVED_PREDICTIONS_AUDIT", "test_used": False,
              "input_domain": "real observed AGRI", "validation_images": len(val_rows), "arms": {}}
    paths = {"frozen_R": root / "runs/cot_repaired_seed42_v1/validation_predictions.npz",
             "log_control": run / "log_control/validation_predictions.npz",
             "high_cloud_weighted": run / "high_cloud_weighted/validation_predictions.npz"}
    for arm, path in paths.items():
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["row_indices"]
            prediction = saved["pred_log1p_cot"]
            reference = saved["reference_cot"]
            valid = saved["mask"]
        if not np.array_equal(ids, val_rows) or not np.array_equal(reference, target) or not np.array_equal(valid, mask):
            raise RuntimeError(f"{arm} validation rows/CPP target/mask mismatch")
        if prediction.shape != target.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise RuntimeError(f"{arm} prediction shape/nonnegative contract failed")
        all_score = score(prediction, target, mask)
        high = score(prediction, target, mask & (target >= 30))
        station_5x5 = score(prediction, target, mask & center)
        clear = score(prediction, target, mask & (target < 0.1))
        by_station = {name: score(prediction[stations == name], target[stations == name], mask[stations == name])
                      for name in ("sili", "zhujia")}
        if arm != "frozen_R":
            claimed = complete["control" if arm == "log_control" else "high_cloud_weighted"]
            for key, observed in (("cot_rmse", all_score["rmse"]),
                                  ("high_cot_ge30_rmse", high["rmse"]),
                                  ("station_5x5_rmse", station_5x5["rmse"]),
                                  ("clear_cot_lt0p1_rmse", clear["rmse"])):
                if abs(claimed[key] - observed) > .001:
                    raise RuntimeError(f"{arm}/{key} reported metric drift")
        report["arms"][arm] = {"all": all_score, "cot_ge30": high, "station_5x5": station_5x5,
                               "clear_lt0p1": clear, "by_station": by_station}
    output = run / "independent_saved_prediction_audit.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite independent audit")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"], "arms": {arm: values["all"] for arm, values in report["arms"].items()}},
                     indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.run)
