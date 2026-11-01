"""Audit Hubei COT bank keys/provenance and cross-check validation predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    pack, run, bank = map(Path, (a.pack, a.run, a.bank))
    rep = json.loads((bank / "BANK_REPORT.json").read_text(encoding="utf-8"))
    if rep.get("state") != "HUBEI_FROZEN_R_COT_BANK_CPP_SUPPORTED_SUBSET_ONLY_TEST_CLOSED" or rep.get("test_payloads_read") != 0:
        raise ValueError("Bank is not the expected CPP-supported train/validation subset")
    if sha(run / "best.pt") != rep["frozen_R_checkpoint_sha256"]:
        raise ValueError("COT bank checkpoint provenance mismatch")
    bank_rows = read_rows(bank / "rows.csv")
    pack_rows = read_rows(pack / "rows.csv")
    if len(bank_rows) != len(pack_rows) or any(r["split"] not in ("train", "val") for r in bank_rows):
        raise ValueError("Bank rows missing or contain forbidden split")
    for i, (br, pr) in enumerate(zip(bank_rows, pack_rows)):
        if int(br["index"]) != i or int(pr["index"]) != i or br["time_bjt"] != pr["time_bjt"] or br["split"] != pr["split"]:
            raise ValueError(f"Bank/pack row mismatch at {i}")
    pred = np.load(bank / "cot_pred_trainval.npy", allow_pickle=False)
    if pred.shape != (len(bank_rows), 1, 16, 16) or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("Invalid COT prediction array")
    val_run_rows = read_rows(run / "val_keys.csv")
    val_pred = np.load(run / "val_prediction_cot.npy", allow_pickle=False)
    if len(val_run_rows) != len(val_pred):
        raise ValueError("Saved validation keys/predictions mismatch")
    by_time = {r["time_bjt"]: i for i, r in enumerate(bank_rows)}
    replay = []
    for row, expected in zip(val_run_rows, val_pred):
        idx = by_time.get(row["time_bjt"])
        if idx is None or bank_rows[idx]["split"] != "val":
            raise ValueError("Validation key absent from COT bank")
        replay.append(pred[idx].astype(np.float64))
    replay = np.stack(replay, axis=0)
    max_diff = float(np.max(np.abs(replay - val_pred.astype(np.float64))))
    abs_diff = np.abs(replay - val_pred.astype(np.float64))
    val_truth = np.load(run / "val_reference_cpp_cot.npy", allow_pickle=False).astype(np.float64)
    val_mask = np.load(run / "val_valid_mask.npy", allow_pickle=False)
    if val_truth.shape != val_mask.shape or val_truth.shape != val_pred.shape:
        raise ValueError("Saved validation truth/mask/prediction shapes disagree")

    def metrics(values):
        err = values[val_mask] - val_truth[val_mask]
        y = val_truth[val_mask]
        total = np.sum((y - y.mean(dtype=np.float64)) ** 2, dtype=np.float64)
        return {"rmse_cot": float(np.sqrt(np.mean(err**2, dtype=np.float64))),
                "mae_cot": float(np.mean(np.abs(err), dtype=np.float64)),
                "bias_cot": float(np.mean(err, dtype=np.float64)),
                "r2_cot": float(1.0 - np.sum(err**2, dtype=np.float64) / total) if total > 0 else None}

    saved_val_metrics = metrics(val_pred.astype(np.float64))
    bank_val_metrics = metrics(replay)
    report = {
        "state": "HUBEI_COT_CPP_SUBSET_KEYS_PASSED_VAL_NUMERIC_REPLAY_RECORDED_TEST_CLOSED",
        "test_payloads_read": 0,
        "rows_by_split": rep["rows_by_split"],
        "bank_scope": "CPP-supported supervision subset only; not full AGRI train/validation coverage",
        "prediction_shape": list(pred.shape),
        "train_predictions_are_oof": False,
        "checkpoint_sha256": sha(run / "best.pt"),
        "prediction_sha256": sha(bank / "cot_pred_trainval.npy"),
        "rows_sha256": sha(bank / "rows.csv"),
        "validation_rows_replayed": len(replay),
        "max_abs_difference_vs_saved_validation_predictions": max_diff,
        "mean_abs_difference_vs_saved_validation_predictions": float(abs_diff.mean(dtype=np.float64)),
        "p95_abs_difference_vs_saved_validation_predictions": float(np.quantile(abs_diff, 0.95)),
        "saved_val_metrics": saved_val_metrics,
        "bank_val_metrics": bank_val_metrics,
        "metric_delta_bank_minus_saved": {k: (bank_val_metrics[k] - saved_val_metrics[k])
                                           if saved_val_metrics[k] is not None else None
                                           for k in saved_val_metrics},
        "numeric_note": "Bank and training validation used different batch partitioning under fp16 autocast; report numerical replay differences rather than requiring bitwise equality.",
        "reference_scope": "CPP retrieval consistency only; not independent physical truth or GHI skill",
    }
    Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
