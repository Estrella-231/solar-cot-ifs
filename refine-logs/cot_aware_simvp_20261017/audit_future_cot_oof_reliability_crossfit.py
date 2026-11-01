#!/usr/bin/env python3
"""Fit cross-fitted train-only lead/station COT calibration from OOF forecasts.

The reference is frozen-R COT retrieved from observed training AGRI targets.
Only train rows are indexed from either pack; no validation/test/buffer target
payload or CPP product is used. The resulting affine maps act on future log1p
COT only and are intended as a single-factor downstream GHI input experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
LEADS = tuple(range(15, 241, 15))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_coeff(pred: np.ndarray, truth: np.ndarray, min_pairs: int):
    if len(pred) < min_pairs:
        raise ValueError("insufficient train-only COT pairs for calibration")
    mu_p, mu_y = float(pred.mean()), float(truth.mean())
    var_p = float(np.mean(np.square(pred - mu_p)))
    raw_beta = float(np.mean((pred - mu_p) * (truth - mu_y)) / max(var_p, 1e-12))
    beta = float(np.clip(raw_beta, 0.0, 1.0))
    return {"slope": beta, "unclipped_slope": raw_beta,
            "intercept": mu_y - beta * mu_p,
            "forecast_log1p_mean": mu_p, "reference_log1p_mean": mu_y,
            "n_train_oof_pairs": int(len(pred))}


def fit_reliability(ghi_pack: Path, actual_r_pack: Path, bank_index_path: Path,
                    fold_root: Path, out: Path, min_pairs: int = 1000) -> dict:
    if out.exists():
        raise FileExistsError(out)
    gm = json.loads((ghi_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    rm = json.loads((actual_r_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    if gm.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("forecast GHI pack is incomplete")
    if gm.get("train_forecast_oof") is not True or gm.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("COT calibration requires five-fold OOF training forecasts")
    if gm.get("test_payload_opened") is not False or gm.get("buffer_payload_opened") is not False:
        raise ValueError("forecast GHI pack does not certify test/buffer closure")
    if gm.get("future_observed_agri_used_as_input") is not False or gm.get("cpp_used_as_input") is not False:
        raise ValueError("forecast GHI pack violates causal inference inputs")
    if rm.get("state") != "COMPLETE" or rm.get("split_id") != gm.get("split_id"):
        raise ValueError("actual-R pack state/split differs from forecast pack")
    if rm.get("test_payload_opened") is not False:
        raise ValueError("actual-R pack does not certify test closure")

    # Bind each GHI row's sequence index to the exact date-heldout OOF fold.
    seq_by_index = {}
    with bank_index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            idx = int(row["sequence_index_within_split"])
            if idx in seq_by_index:
                raise ValueError("duplicate train sequence index")
            seq_by_index[idx] = row["seq_id"]
    fold_by_seq = {}
    fold_hashes = {}
    for fold in range(5):
        ids_path = fold_root / ("fold_%d" % fold) / "fold_oof_ids.txt"
        if not ids_path.is_file():
            raise FileNotFoundError(ids_path)
        fold_hashes[str(fold)] = sha(ids_path)
        for seq_id in ids_path.read_text(encoding="utf-8").splitlines():
            if seq_id in fold_by_seq:
                raise ValueError("sequence assigned to more than one heldout fold")
            fold_by_seq[seq_id] = fold
    if set(fold_by_seq) != set(seq_by_index.values()):
        raise ValueError("five heldout-fold IDs do not exactly cover train OOF sequences")

    actual_index = {}
    with (actual_r_pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            if row["split"] != "train":
                continue
            key = (row["station"], row["target_time_utc"])
            if key in actual_index:
                raise ValueError("duplicate actual-R train key: %r" % (key,))
            actual_index[key] = idx

    train_dir = ghi_pack / "train"
    keys_path = train_dir / "keys.csv"
    rows = []
    key_seen = set(); forecast_target_seen = set()
    with keys_path.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row.get("split") != "train":
                raise ValueError("non-train metadata appeared in the train pack")
            station = row["station"]
            lead = int(row["lead_minutes"])
            if station not in STATIONS or lead not in LEADS:
                raise ValueError("unexpected station/lead in forecast pack")
            key = (station, row["init_time_utc"], row["target_time_utc"], lead,
                   int(row["ghi_row_index"]))
            if key in key_seen:
                raise ValueError("duplicate GHI train row key")
            key_seen.add(key)
            target_key = (station, row["target_time_utc"], lead)
            if target_key in forecast_target_seen:
                raise ValueError("duplicate station/target/lead forecast pair")
            forecast_target_seen.add(target_key)
            ref_idx = actual_index.get((station, row["target_time_utc"]))
            if ref_idx is None:
                raise ValueError("missing train-only actual-R COT reference for %r" %
                                 ((station, row["target_time_utc"]),))
            seq_index = int(row["sequence_index_within_split"])
            seq_id = seq_by_index.get(seq_index)
            if seq_id is None or seq_id not in fold_by_seq:
                raise ValueError("GHI training row cannot be assigned to a heldout OOF fold")
            rows.append((i, station, lead, ref_idx, fold_by_seq[seq_id]))
    if len(rows) != gm.get("rows", {}).get("train") or not rows:
        raise ValueError("unexpected forecast train row count")

    forecast = np.load(train_dir / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    actual = np.load(actual_r_pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if forecast.shape != (len(rows), 3, 1, 3, 3) or forecast.dtype != np.float32:
        raise ValueError("unexpected OOF forecast-COT pack shape/dtype")
    if actual.shape[0] != len(list(csv.DictReader((actual_r_pack / "rows.csv").open(newline="", encoding="utf-8")))) \
            or actual.ndim != 4 or actual.shape[1:] != (1, 3, 3) or actual.dtype != np.float32:
        raise ValueError("unexpected actual-R COT target pack shape/dtype")
    if sha(train_dir / "cot_log1p.npy") != gm.get("array_sha256", {}).get("train", {}).get("cot_log1p.npy"):
        raise ValueError("OOF forecast COT array hash mismatch")
    if sha(actual_r_pack / "cot_log1p.npy") != rm.get("array_sha256", {}).get("cot_log1p.npy"):
        raise ValueError("actual-R train reference array hash mismatch")
    if sha(actual_r_pack / "rows.csv") != rm.get("array_sha256", {}).get("rows.csv"):
        raise ValueError("actual-R row index hash mismatch")

    grouped = {(s, lead, fold): [[], []] for s in STATIONS for lead in LEADS for fold in range(5)}
    for i, station, lead, ref_idx, fold in rows:
        pred = float(forecast[i, 2, 0, 1, 1])
        truth = float(actual[ref_idx, 0, 1, 1])
        if not (np.isfinite(pred) and np.isfinite(truth)):
            raise ValueError("nonfinite train-only COT pair")
        grouped[(station, lead, fold)][0].append(pred)
        grouped[(station, lead, fold)][1].append(truth)

    params_full, params_excluded, summary = {}, {str(k): {} for k in range(5)}, []
    for station in STATIONS:
        for lead in LEADS:
            fold_data = {k: (np.asarray(grouped[(station, lead, k)][0], np.float64),
                             np.asarray(grouped[(station, lead, k)][1], np.float64))
                         for k in range(5)}
            pred = np.concatenate([fold_data[k][0] for k in range(5)])
            truth = np.concatenate([fold_data[k][1] for k in range(5)])
            if len(pred) < min_pairs:
                raise ValueError("insufficient train OOF pairs for %s/%s" % (station, lead))
            full = fit_coeff(pred, truth, min_pairs)
            params_full.setdefault(station, {})[str(lead)] = full
            rmse = float(np.sqrt(np.mean(np.square(pred - truth))))
            full_calibrated = full["intercept"] + full["slope"] * pred
            full_rmse = float(np.sqrt(np.mean(np.square(full_calibrated - truth))))
            base_rmse = float(np.sqrt(np.mean(np.square(truth - float(truth.mean())))))
            corr = (float(np.corrcoef(pred, truth)[0, 1])
                    if len(pred) >= 2 and np.std(pred) > 0 and np.std(truth) > 0 else None)
            bias = float(np.mean(pred - truth))
            cf_pred, cf_truth = [], []
            for excluded in range(5):
                fit_p = np.concatenate([fold_data[k][0] for k in range(5) if k != excluded])
                fit_y = np.concatenate([fold_data[k][1] for k in range(5) if k != excluded])
                coef = fit_coeff(fit_p, fit_y, min_pairs)
                params_excluded[str(excluded)].setdefault(station, {})[str(lead)] = coef
                held_p, held_y = fold_data[excluded]
                if len(held_p):
                    cf_pred.extend((coef["intercept"] + coef["slope"] * held_p).tolist())
                    cf_truth.extend(held_y.tolist())
            cf_rmse = float(np.sqrt(np.mean(np.square(np.asarray(cf_pred) - np.asarray(cf_truth)))))
            summary.append({"station": station, "lead_minutes": lead,
                "n_train_oof_pairs": int(len(pred)), "correlation": corr,
                "unclipped_slope": full["unclipped_slope"], "slope": full["slope"],
                "intercept": full["intercept"],
                "bias_log1p": bias, "rmse_log1p": rmse,
                "full_fit_calibrated_rmse_log1p_descriptive": full_rmse,
                "fold_excluded_crossfit_calibrated_rmse_log1p": cf_rmse,
                "group_mean_baseline_rmse_log1p": base_rmse,
                "skill_vs_group_mean": 1.0 - rmse / max(base_rmse, 1e-12)})
    out.mkdir(parents=True)
    result = {
        "state": "COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION",
        "scope": "train-only station/lead affine calibration; each GHI train row is transformed by coefficients fitted on the other four OOF folds; validation coefficients use all five train folds",
        "split_id": gm["split_id"], "train_forecast_oof": True,
        "forecast_bank_oof_fold_count": 5,
        "reference": "center pixel of frozen-R COT retrieved from observed future AGRI on training dates only",
        "transform": "future_log1p_COT_cal = intercept[station,lead] + slope[station,lead] * forecast_log1p_COT; apply to all 3x3 pixels at that future frame; leave observed-history COT unchanged",
        "slope_clip": [0.0, 1.0], "fit_pixel": "station center pixel only",
        "train_rows": len(rows), "unique_actual_train_keys": len(actual_index),
        "unmatched_train_rows": 0, "test_payload_opened": False,
        "buffer_payload_opened": False, "cpp_used": False,
        "future_observed_agri_used_as_inference_input": False,
        "observed_future_agri_used_only_for_train_reference_labels": True,
        "forecast_pack_manifest_sha256": sha(ghi_pack / "pack_manifest.json"),
        "actual_r_pack_manifest_sha256": sha(actual_r_pack / "pack_manifest.json"),
        "train_keys_sha256": sha(keys_path),
        "bank_index_sha256": sha(bank_index_path), "fold_oof_ids_sha256": fold_hashes,
        "fold_by_sequence_index": {str(idx): fold_by_seq[seq_id]
                                    for idx, seq_id in seq_by_index.items()},
        "forecast_cot_array_sha256": gm["array_sha256"]["train"]["cot_log1p.npy"],
        "actual_cot_array_sha256": sha(actual_r_pack / "cot_log1p.npy"),
        "parameters_full_train": params_full,
        "parameters_excluding_each_fold": params_excluded, "metrics": summary,
    }
    (out / "calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-pack", type=Path, required=True)
    parser.add_argument("--actual-r-pack", type=Path, required=True)
    parser.add_argument("--bank-index", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=1000)
    args = parser.parse_args()
    result = fit_reliability(args.forecast_pack, args.actual_r_pack, args.bank_index,
                             args.fold_root, args.output,
                             min_pairs=args.min_pairs)
    print(json.dumps({"state": result["state"], "train_rows": result["train_rows"],
        "unmatched_train_rows": result["unmatched_train_rows"],
        "test_payload_opened": result["test_payload_opened"],
        "buffer_payload_opened": result["buffer_payload_opened"],
        "crossfit_train_rows": result["train_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
