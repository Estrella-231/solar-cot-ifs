"""Independently recompute saved forecast-domain COT pilot validation metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


def score(pred_log, reference, mask):
    err = (np.expm1(pred_log.astype(np.float64)) - reference.astype(np.float64))[mask]
    if len(err) == 0:
        raise RuntimeError("empty validation mask")
    return {"pixels": int(len(err)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "bias": float(err.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()
    root, run, pack = args.root.resolve(), args.run.resolve(), args.pack.resolve()
    complete = json.loads((run / "complete.json").read_text())
    if complete["state"] != "COMPLETE_FORECAST_DOMAIN_COT_PAIRED_PILOT" or complete["test_used"] is not False:
        raise RuntimeError("training output incomplete")
    expected_keys = np.load(pack / "val/keys.npy")
    original = root / "data/cot_repaired_pack_20260907_v1"
    original_target = np.load(original / "target.npy", mmap_mode="r")
    original_mask = np.load(original / "mask.npy", mmap_mode="r")
    truth = np.asarray(original_target[expected_keys[:, 3]], dtype=np.float32) * np.float32(100)
    mask = np.asarray(original_mask[expected_keys[:, 3]], dtype=bool)
    center = np.zeros((1, 1, 16, 16), dtype=bool)
    center[:, :, 6:11, 6:11] = True
    report = {"state": "PASS_FORECAST_DOMAIN_COT_SAVED_PREDICTIONS_AUDIT",
              "test_used": False, "validation_pairs": len(expected_keys),
              "unique_cpp_targets": len(np.unique(expected_keys[:, 3])), "arms": {}}
    for arm in ("real_train_control", "forecast_train_adapt"):
        with np.load(run / arm / "validation_predictions.npz", allow_pickle=False) as saved:
            keys = saved["keys"]
            pred_log = saved["pred_log1p_cot"]
            saved_truth = saved["reference_cot"]
            saved_mask = saved["mask"]
        if not np.array_equal(keys, expected_keys) or not np.array_equal(saved_truth, truth) or not np.array_equal(saved_mask, mask):
            raise RuntimeError("arm source/reference/mask mismatch")
        if pred_log.shape != truth.shape or not np.isfinite(pred_log).all() or (pred_log < 0).any():
            raise RuntimeError("invalid arm COT prediction")
        all_score = score(pred_log, truth, mask)
        high_score = score(pred_log, truth, mask & (truth >= 30))
        center_score = score(pred_log, truth, mask & center)
        clear_score = score(pred_log, truth, mask & (truth < 0.1))
        claimed = complete["arms"][arm]
        for key, measured in (("cot_rmse", all_score["rmse"]),
                              ("high_cot_ge30_rmse", high_score["rmse"]),
                              ("station_5x5_rmse", center_score["rmse"]),
                              ("clear_cot_lt0p1_rmse", clear_score["rmse"])):
            if abs(claimed[key] - measured) > 0.001:
                raise RuntimeError(f"saved prediction metric drift: {arm}/{key}/{claimed[key]}/{measured}")
        by_lead = {}
        for lead in range(16):
            chosen = expected_keys[:, 2] == lead
            if chosen.any():
                by_lead[str((lead + 1) * 15)] = score(pred_log[chosen], truth[chosen], mask[chosen])
        by_station = {}
        for station, name in ((0, "Sili"), (1, "Zhujia")):
            chosen = expected_keys[:, 1] == station
            by_station[name] = score(pred_log[chosen], truth[chosen], mask[chosen])
        report["arms"][arm] = {"best_epoch": claimed["best_epoch"], "all": all_score,
                               "cot_ge30": high_score, "station_5x5": center_score,
                               "clear_lt0p1": clear_score, "by_lead_minutes": by_lead,
                               "by_station": by_station}
    output = run / "independent_saved_prediction_audit.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite independent audit")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"],
                      "validation_pairs": report["validation_pairs"],
                      "arms": {key: value["all"] for key, value in report["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
