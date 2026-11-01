#!/usr/bin/env python3
"""Validation-only diagnostic linking SimVP AGRI forecast error to frozen-R COT error.

The observed future AGRI frames are opened only as held-out diagnostic references.
They are never supplied to a deployable inference path, used to fit a model, or
used to select a threshold. Test and buffer frames are not opened.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
LEAD_BANDS = ((15, 60, "15-60"), (75, 150, "75-150"), (165, 240, "165-240"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def corr(x, y) -> float:
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.std(x[good]) == 0 or np.std(y[good]) == 0:
        return float("nan")
    return float(np.corrcoef(x[good], y[good])[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simvp-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--forecast-pack", type=Path, required=True)
    p.add_argument("--oracle-pack", type=Path, required=True)
    p.add_argument("--actual-r-bank", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-frames", type=int, default=0, help="smoke only; never a result")
    a = p.parse_args()
    if a.output.exists() or Path(str(a.output) + ".csv").exists():
        raise FileExistsError("refusing to overwrite existing audit output")
    if a.workers < 1:
        raise ValueError("workers must be positive")

    fm = json.loads((a.forecast_pack / "pack_manifest.json").read_text())
    om = json.loads((a.oracle_pack / "pack_manifest.json").read_text())
    rm = json.loads((a.actual_r_bank / "bank_manifest.json").read_text())
    cal = json.loads(a.calibration.read_text())
    if fm.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK" or fm.get("train_forecast_oof") is not True or fm.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("five-fold OOF forecast pack required")
    if any(fm.get(k) is not False for k in ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input")):
        raise ValueError("forecast inference closure failed")
    if om.get("state") != "COMPLETE_ORACLE_FUTURE_COT_GHI_PACK" or om.get("split_id") != fm.get("split_id") or om.get("deployable") is not False or om.get("test_payloads_read") != 0 or om.get("buffer_payload_opened") is not False:
        raise ValueError("oracle diagnostic source/closure mismatch")
    if rm.get("state") != "COMPLETE" or rm.get("split_id") != fm.get("split_id") or rm.get("test_payload_opened") is not False:
        raise ValueError("actual-R frame index source/closure mismatch")
    if cal.get("state") != "COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION":
        raise ValueError("expected frozen train-only reliability calibration")
    if cal.get("forecast_pack_manifest_sha256") != sha(a.forecast_pack / "pack_manifest.json"):
        raise ValueError("calibration is bound to another forecast pack")

    norm_path = a.simvp_root / "manifests_v1" / "hunan_train_norm.json"
    norm = json.loads(norm_path.read_text())
    if norm.get("validation_test_used") is not False:
        raise ValueError("AGRI normalization lacks train-only provenance")
    agri_std = np.asarray(norm["std"], np.float64)
    if agri_std.shape != (13,) or not np.isfinite(agri_std).all() or (agri_std <= 0).any():
        raise ValueError("invalid 13-channel train-only AGRI std")

    sys.path.insert(0, str(a.simvp_root / "code"))
    from hunan_data import load_frame
    from train_future_ghi_heads_cot_reliability import apply_reliability_calibration

    keys = read_csv(a.forecast_pack / "val" / "keys.csv")
    oracle_keys = read_csv(a.oracle_pack / "val" / "keys.csv")
    if len(keys) != 66590 or len(oracle_keys) != len(keys):
        raise ValueError("unexpected validation cohort size")
    if keys != oracle_keys:
        raise ValueError("forecast/oracle validation keys differ")
    pred_agri = np.load(a.forecast_pack / "val" / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    pred_cot = np.load(a.forecast_pack / "val" / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    ref_cot = np.load(a.oracle_pack / "val" / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if pred_agri.shape != (len(keys), 3, 13, 3, 3) or pred_agri.dtype != np.float32:
        raise ValueError("forecast AGRI validation array shape/dtype mismatch")
    if pred_cot.shape != (len(keys), 3, 1, 3, 3) or ref_cot.shape != pred_cot.shape or pred_cot.dtype != np.float32 or ref_cot.dtype != np.float32:
        raise ValueError("COT validation array shape/dtype mismatch")
    cal_pred_cot = apply_reliability_calibration(np.asarray(pred_cot), keys, "val", cal)

    frame_rows = [r for r in read_csv(a.actual_r_bank / "frame_keys.csv") if r.get("split") == "val"]
    frame_meta = {}
    for r in frame_rows:
        ts = r["target_time_utc"]
        if ts in frame_meta:
            raise ValueError(f"duplicate validation frame metadata: {ts}")
        frame_meta[ts] = {"relpath": r["agri_frame_relpath"],
                          **{s: tuple(map(int, r[f"{s}_grid_rc"].split(","))) for s in STATIONS}}
    target_times = sorted({r["target_time_utc"] for r in keys})
    if not set(target_times) <= set(frame_meta):
        raise ValueError("validation target frames missing actual-R station metadata")
    if a.limit_frames:
        target_times = target_times[:a.limit_frames]
        allowed = set(target_times)
        keep = np.asarray([r["target_time_utc"] in allowed for r in keys])
        keys = [r for r, ok in zip(keys, keep) if ok]
        pred_agri = np.asarray(pred_agri)[keep]
        pred_cot = np.asarray(pred_cot)[keep]
        ref_cot = np.asarray(ref_cot)[keep]
        cal_pred_cot = cal_pred_cot[keep]
        smoke = True
    else:
        smoke = False

    def load_patches(ts):
        meta = frame_meta[ts]
        image, valid, _ = load_frame(a.data_root, meta["relpath"])
        result = {}
        for s in STATIONS:
            gr, gc = meta[s]
            r0, c0 = gr - 1, gc - 1
            patch = image[:, r0:r0 + 3, c0:c0 + 3]
            mask = valid[:, r0:r0 + 3, c0:c0 + 3]
            if patch.shape != (13, 3, 3) or mask.shape != patch.shape:
                raise ValueError(f"invalid station patch at {ts}/{s}")
            result[s] = (patch.astype(np.float32, copy=True), mask.astype(bool, copy=True))
        return ts, result

    actual = {}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, (ts, patches) in enumerate(pool.map(load_patches, target_times), start=1):
            actual[ts] = patches
            if i % 500 == 0:
                print(f"ACTUAL_AGRI_VALIDATION_FRAMES {i}/{len(target_times)}", flush=True)

    ch_err2 = {s: np.zeros((16, 13), np.float64) for s in STATIONS}
    ch_bias = {s: np.zeros((16, 13), np.float64) for s in STATIONS}
    ch_pred_sum = {s: np.zeros((16, 13), np.float64) for s in STATIONS}
    ch_true_sum = {s: np.zeros((16, 13), np.float64) for s in STATIONS}
    ch_pred_sq = {s: np.zeros((16, 13), np.float64) for s in STATIONS}
    ch_true_sq = {s: np.zeros((16, 13), np.float64) for s in STATIONS}
    ch_count = {s: np.zeros((16, 13), np.int64) for s in STATIONS}
    per_row = []
    agri_err_all = []; cot_abs_all = []; cot_err_all = []
    for i, row in enumerate(keys):
        s = row["station"].lower(); lead = int(row["lead_minutes"])
        if s not in STATIONS or lead not in range(15, 241, 15):
            raise ValueError("unexpected station/lead in validation keys")
        slot = 2
        tru, valid = actual[row["target_time_utc"]][s]
        pred = np.asarray(pred_agri[i, slot], np.float32)
        err = (pred.astype(np.float64) - tru.astype(np.float64)) / agri_std[:, None, None]
        nrmse = np.full(13, np.nan, np.float64)
        bias = np.full(13, np.nan, np.float64)
        for c in range(13):
            m = valid[c]
            if m.any():
                ec = err[c][m]
                nrmse[c] = np.sqrt(np.mean(ec * ec))
                bias[c] = np.mean(ec)
                ch_err2[s][lead // 15 - 1, c] += float(np.sum(ec * ec))
                ch_bias[s][lead // 15 - 1, c] += float(np.sum(ec))
                ch_count[s][lead // 15 - 1, c] += int(m.sum())
                ch_pred_sum[s][lead // 15 - 1, c] += float(np.sum(pred[c][m] / agri_std[c]))
                ch_true_sum[s][lead // 15 - 1, c] += float(np.sum(tru[c][m] / agri_std[c]))
                ch_pred_sq[s][lead // 15 - 1, c] += float(np.sum((pred[c][m] / agri_std[c]) ** 2))
                ch_true_sq[s][lead // 15 - 1, c] += float(np.sum((tru[c][m] / agri_std[c]) ** 2))
        sample_agri = float(np.sqrt(np.nanmean(nrmse * nrmse)))
        raw_cot_err = np.asarray(pred_cot[i, slot, 0] - ref_cot[i, slot, 0], np.float64)
        cal_cot_err = np.asarray(cal_pred_cot[i, slot, 0] - ref_cot[i, slot, 0], np.float64)
        cot_center_abs = float(abs(cal_cot_err[1, 1]))
        cot_center_signed = float(cal_cot_err[1, 1])
        cot_patch_rmse = float(np.sqrt(np.mean(cal_cot_err * cal_cot_err)))
        cot_raw_patch_rmse = float(np.sqrt(np.mean(raw_cot_err * raw_cot_err)))
        agri_err_all.append(sample_agri); cot_abs_all.append(cot_center_abs); cot_err_all.append(cot_center_signed)
        per_row.append((s, row["target_time_utc"], lead, sample_agri, cot_center_abs, cot_center_signed, cot_patch_rmse, cot_raw_patch_rmse, *nrmse.tolist()))

    channel_summary = []
    for s in STATIONS:
        for lead in range(15, 241, 15):
            ti = lead // 15 - 1
            for c in range(13):
                n = int(ch_count[s][ti, c])
                if not n:
                    continue
                pred_mean = ch_pred_sum[s][ti, c] / n
                true_mean = ch_true_sum[s][ti, c] / n
                pred_var = max(ch_pred_sq[s][ti, c] / n - pred_mean * pred_mean, 0.0)
                true_var = max(ch_true_sq[s][ti, c] / n - true_mean * true_mean, 0.0)
                channel_summary.append({"station": s, "lead_minutes": lead, "channel_index_0based": c,
                    "valid_pixels": n, "normalized_rmse": float(np.sqrt(ch_err2[s][ti, c] / n)),
                    "normalized_bias": float(ch_bias[s][ti, c] / n),
                    "forecast_to_observed_std_ratio": float(np.sqrt(pred_var / true_var)) if true_var > 0 else None})

    agri_err_all = np.asarray(agri_err_all); cot_abs_all = np.asarray(cot_abs_all); cot_err_all = np.asarray(cot_err_all)
    station_metrics = {}
    for s in STATIONS:
        pick = np.asarray([r[0] == s for r in per_row])
        station_metrics[s] = {"rows": int(pick.sum()),
            "mean_patch_agri_channel_standardized_rmse": float(np.mean(agri_err_all[pick])),
            "mean_abs_center_calibrated_cot_error_log1p": float(np.mean(cot_abs_all[pick])),
            "pearson_agri_rmse_vs_abs_center_cot_error": corr(agri_err_all[pick], cot_abs_all[pick]),
            "pearson_agri_rmse_vs_signed_center_cot_error": corr(agri_err_all[pick], cot_err_all[pick])}
    channel_corr = {}
    matrix = np.asarray([r[8:21] for r in per_row], np.float64)
    for c in range(13):
        channel_corr[str(c)] = corr(matrix[:, c], cot_abs_all)

    bands = []
    for s in STATIONS:
        ss = np.asarray([r[0] == s for r in per_row])
        for lo, hi, name in LEAD_BANDS:
            q = ss & np.asarray([lo <= int(r[2]) <= hi for r in per_row])
            bands.append({"station": s, "lead_band_minutes": name, "rows": int(q.sum()),
                "mean_agri_standardized_rmse": float(np.mean(agri_err_all[q])),
                "mean_abs_center_calibrated_cot_error_log1p": float(np.mean(cot_abs_all[q])),
                "mean_signed_center_calibrated_cot_error_log1p": float(np.mean(cot_err_all[q])),
                "pearson_agri_rmse_vs_abs_center_cot_error": corr(agri_err_all[q], cot_abs_all[q])})

    out = {"state": "COMPLETE_VALIDATION_FORECAST_AGRI_TO_FROZEN_R_DOMAIN_DIAGNOSTIC_SMOKE" if smoke else "COMPLETE_VALIDATION_FORECAST_AGRI_TO_FROZEN_R_DOMAIN_DIAGNOSTIC",
        "question": "whether errors in predicted AGRI at the station neighborhood co-vary with downstream frozen-R future-COT error",
        "scope": "validation-only descriptive audit; no fitting, thresholds, GHI labels, or model selection",
        "rows": int(len(keys)), "unique_observed_validation_frames_opened": int(len(actual)), "stations": list(STATIONS),
        "lead_bands": [{"label": x[2], "min": x[0], "max": x[1]} for x in LEAD_BANDS],
        "station_metrics": station_metrics, "lead_band_metrics": bands,
        "pearson_channel_standardized_agri_error_vs_abs_center_calibrated_cot_error": channel_corr,
        "channel_metrics": channel_summary,
        "forecast_pack_manifest_sha256": sha(a.forecast_pack / "pack_manifest.json"),
        "forecast_agri_val_sha256": sha(a.forecast_pack / "val" / "agri_raw.npy"),
        "forecast_cot_val_sha256": sha(a.forecast_pack / "val" / "cot_log1p.npy"),
        "oracle_pack_manifest_sha256": sha(a.oracle_pack / "pack_manifest.json"),
        "oracle_cot_val_sha256": sha(a.oracle_pack / "val" / "cot_log1p.npy"),
        "actual_r_bank_manifest_sha256": sha(a.actual_r_bank / "bank_manifest.json"),
        "actual_r_frame_keys_sha256": sha(a.actual_r_bank / "frame_keys.csv"),
        "train_only_agri_norm_sha256": sha(norm_path), "calibration_sha256": sha(a.calibration),
        "observed_future_agri_used_as_diagnostic_reference": True,
        "observed_future_agri_used_as_inference_input": False, "cpp_used": False,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "validation_used_for_fitting_or_threshold_selection": False,
        "r_is_retrieval_reference_not_independent_cloud_truth": True,
        "smoke_limit_frames": int(a.limit_frames)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with Path(str(a.output) + ".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("station", "target_time_utc", "lead_minutes", "agri_patch_nrmse", "calibrated_cot_center_abs_error_log1p", "calibrated_cot_center_signed_error_log1p", "calibrated_cot_patch_rmse_log1p", "raw_cot_patch_rmse_log1p", *[f"agri_channel_{c}_nrmse" for c in range(13)])); w.writerows(per_row)
    print(json.dumps({"state": out["state"], "rows": out["rows"], "frames": out["unique_observed_validation_frames_opened"], "station_metrics": station_metrics, "lead_band_metrics": bands, "test_payload_opened": False, "buffer_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
