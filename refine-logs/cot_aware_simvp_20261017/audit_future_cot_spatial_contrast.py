#!/usr/bin/env python3
"""Fit and cross-validate a train-only 3x3 forecast-COT contrast calibration.

Only train OOF forecast COT and frozen-R COT retrieved from observed training
AGRI are used to fit the center-preserving neighbor-contrast coefficient. No
validation labels, CPP, test, or buffer payload is opened by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STATIONS = ("sili", "zhujia")
LEADS = tuple(range(15, 241, 15))
NEIGHBOR = np.ones((3, 3), dtype=bool); NEIGHBOR[1, 1] = False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_gamma(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    den = float(np.sum(np.square(x), dtype=np.float64))
    raw = float(np.sum(x * y, dtype=np.float64) / max(den, 1e-12))
    return float(np.clip(raw, 0.0, 2.0)), raw, den


def balanced_patch_rmse(err_by_group: dict[tuple[str, int], np.ndarray]) -> float:
    return float(np.mean([np.sqrt(np.mean(e)) for e in err_by_group.values() if e.size]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-pack", type=Path, required=True)
    p.add_argument("--actual-r-pack", type=Path, required=True)
    p.add_argument("--bank-index", type=Path, required=True)
    p.add_argument("--fold-root", type=Path, required=True)
    p.add_argument("--center-calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    fm = json.loads((args.forecast_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    rm = json.loads((args.actual_r_pack / "pack_manifest.json").read_text(encoding="utf-8"))
    cc = json.loads(args.center_calibration.read_text(encoding="utf-8"))
    if fm.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK" or fm.get("train_forecast_oof") is not True:
        raise ValueError("forecast pack must be complete and five-fold OOF")
    if fm.get("forecast_bank_oof_fold_count") != 5 or fm.get("test_payload_opened") is not False or fm.get("buffer_payload_opened") is not False:
        raise ValueError("forecast pack OOF/test/buffer provenance gate failed")
    if fm.get("future_observed_agri_used_as_input") is not False or fm.get("cpp_used_as_input") is not False:
        raise ValueError("noncausal inference feature declared in pack")
    if rm.get("state") != "COMPLETE" or rm.get("split_id") != fm.get("split_id") or rm.get("test_payload_opened") is not False:
        raise ValueError("actual-R reference pack provenance mismatch")
    if cc.get("state") != "COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION":
        raise ValueError("center calibration is not the registered train-only calibration")
    if cc.get("forecast_pack_manifest_sha256") != sha(args.forecast_pack / "pack_manifest.json"):
        raise ValueError("center calibration references a different forecast pack")

    seq_by_idx = {}
    with args.bank_index.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                ix = int(r["sequence_index_within_split"])
                if ix in seq_by_idx: raise ValueError("duplicate bank sequence index")
                seq_by_idx[ix] = r["seq_id"]
    seq_fold = {}
    fold_hashes = {}
    for fold in range(5):
        fp = args.fold_root / f"fold_{fold}" / "fold_oof_ids.txt"
        fold_hashes[str(fold)] = sha(fp)
        for seq in fp.read_text(encoding="utf-8").splitlines():
            if seq in seq_fold: raise ValueError("sequence appears in multiple heldout folds")
            seq_fold[seq] = fold
    if set(seq_fold) != set(seq_by_idx.values()): raise ValueError("fold IDs do not exactly cover train OOF bank")

    actual_rows = list(csv.DictReader((args.actual_r_pack / "rows.csv").open(newline="", encoding="utf-8")))
    actual_index = {}
    for i, r in enumerate(actual_rows):
        if r["split"] == "train":
            k = (r["station"], r["target_time_utc"])
            if k in actual_index: raise ValueError("duplicate actual-R train reference key")
            actual_index[k] = i

    keys = list(csv.DictReader((args.forecast_pack / "train" / "keys.csv").open(newline="", encoding="utf-8")))
    group_ids: dict[tuple[str, int, int], list[int]] = {}
    refs = np.empty(len(keys), np.int64); folds = np.empty(len(keys), np.int8); days = []
    seen = set()
    for i, r in enumerate(keys):
        if r["split"] != "train": raise ValueError("non-train row in train keys")
        st, lead = r["station"], int(r["lead_minutes"])
        if st not in STATIONS or lead not in LEADS: raise ValueError("unexpected station/lead")
        k = (st, r["init_time_utc"], r["target_time_utc"], lead, int(r["ghi_row_index"]))
        if k in seen: raise ValueError("duplicate forecast GHI train key")
        seen.add(k)
        ref = actual_index.get((st, r["target_time_utc"]))
        if ref is None: raise ValueError("forecast row missing same-time train R reference")
        seq_ix = int(r["sequence_index_within_split"])
        fold = seq_fold.get(seq_by_idx.get(seq_ix, ""))
        if fold is None: raise ValueError("train row missing OOF fold assignment")
        refs[i], folds[i] = ref, fold
        days.append((datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d"))
        group_ids.setdefault((st, lead, fold), []).append(i)
    if len(keys) != fm.get("rows", {}).get("train") or not keys: raise ValueError("unexpected train row coverage")

    pred = np.load(args.forecast_pack / "train" / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(args.actual_r_pack / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    if pred.shape != (len(keys), 3, 1, 3, 3) or pred.dtype != np.float32: raise ValueError("forecast COT shape/dtype mismatch")
    if truth.shape != (len(actual_rows), 1, 3, 3) or truth.dtype != np.float32: raise ValueError("actual-R COT shape/dtype mismatch")
    if sha(args.forecast_pack / "train" / "cot_log1p.npy") != fm["array_sha256"]["train"]["cot_log1p.npy"]: raise ValueError("forecast COT hash mismatch")
    if sha(args.actual_r_pack / "cot_log1p.npy") != rm["array_sha256"]["cot_log1p.npy"]: raise ValueError("actual-R COT hash mismatch")
    if sha(args.actual_r_pack / "rows.csv") != rm["array_sha256"]["rows.csv"]: raise ValueError("actual-R rows hash mismatch")

    # Fold-specific sums are sufficient statistics for train-excluded gamma.
    sums: dict[tuple[str, int, int], tuple[float, float]] = {}
    full_sums: dict[tuple[str, int], tuple[float, float]] = {}
    prepared = {}
    for (st, lead, fold), ix_list in group_ids.items():
        ix = np.asarray(ix_list, np.int64); y = np.asarray(truth[refs[ix], 0], np.float64)
        pp = np.asarray(pred[ix, 2, 0], np.float64)
        co = cc["parameters_excluding_each_fold"][str(fold)][st][str(lead)]
        pc = float(co["intercept"]) + float(co["slope"]) * pp
        xc = pc[:, 1, 1]
        yc = y[:, 1, 1]
        xdev = pc - xc[:, None, None]; ydev = y - yc[:, None, None]
        x = xdev[:, NEIGHBOR].reshape(-1); yy = ydev[:, NEIGHBOR].reshape(-1)
        sx = float(np.sum(x * yy, dtype=np.float64)); ss = float(np.sum(x * x, dtype=np.float64))
        sums[(st, lead, fold)] = (sx, ss)
        prepared[(st, lead, fold)] = (ix, pp, y, pc, xdev, ydev)
        full_coef = cc["parameters_full_train"][st][str(lead)]
        full_pc = float(full_coef["intercept"]) + float(full_coef["slope"]) * pp
        full_xdev = full_pc - full_pc[:,1,1,None,None]
        fsx = float(np.sum(full_xdev[:,NEIGHBOR] * ydev[:,NEIGHBOR], dtype=np.float64))
        fss = float(np.sum(np.square(full_xdev[:,NEIGHBOR]), dtype=np.float64))
        old = full_sums.get((st,lead),(0.0,0.0))
        full_sums[(st,lead)] = (old[0]+fsx,old[1]+fss)

    full_gamma, excluded_gamma, metrics, row_delta = {}, {str(k): {} for k in range(5)}, [], np.asarray(days)
    # Store per-row center+patch MSE for date-block paired uncertainty.
    base_mse = np.zeros(len(keys), np.float64); new_mse = np.zeros(len(keys), np.float64)
    for st in STATIONS:
        for lead in LEADS:
            present = [f for f in range(5) if (st, lead, f) in sums]
            if len(present) != 5: raise ValueError(f"missing fold coverage at {st}/{lead}")
            sx, ss = full_sums[(st,lead)]
            raw = sx / max(ss, 1e-12); g = float(np.clip(raw, 0.0, 2.0))
            full_gamma.setdefault(st, {})[str(lead)] = {"gamma": g, "unclipped_gamma": float(raw), "n_oof_sequences": int(sum(len(prepared[(st,lead,f)][0]) for f in range(5)))}
            for excluded in range(5):
                sx_e = sum(sums[(st, lead, f)][0] for f in present if f != excluded)
                ss_e = sum(sums[(st, lead, f)][1] for f in present if f != excluded)
                ge_raw = sx_e / max(ss_e, 1e-12); ge = float(np.clip(ge_raw, 0.0, 2.0))
                excluded_gamma[str(excluded)].setdefault(st, {})[str(lead)] = {"gamma": ge, "unclipped_gamma": float(ge_raw)}
            for fold in range(5):
                ix, pp, y, pc, xdev, ydev = prepared[(st, lead, fold)]
                ge = excluded_gamma[str(fold)][st][str(lead)]["gamma"]
                cand = pc[:,1,1,None,None] + ge*xdev
                base_mse[ix] = np.mean(np.square(pc-y), axis=(1,2))
                new_mse[ix] = np.mean(np.square(cand-y), axis=(1,2))
                metrics.append({"station": st, "lead_minutes": lead, "fold": fold,
                    "n": int(len(ix)), "gamma_excluding_fold": ge,
                    "center_rmse_log1p_acrel": float(np.sqrt(np.mean((pc[:,1,1]-y[:,1,1])**2))),
                    "center_rmse_log1p_candidate": float(np.sqrt(np.mean((cand[:,1,1]-y[:,1,1])**2))),
                    "patch_rmse_log1p_acrel": float(np.sqrt(np.mean(np.square(pc-y)))),
                    "patch_rmse_log1p_candidate": float(np.sqrt(np.mean(np.square(cand-y))))})
    # Date-block bootstrap on paired train-only OOF reference errors, equal station/lead weighting.
    days_unique = np.asarray(sorted(set(row_delta.tolist())))
    dmap = {d:i for i,d in enumerate(days_unique)}; day_idx = np.asarray([dmap[d] for d in row_delta])
    rng = np.random.default_rng(20261020); draws = np.empty(args.bootstrap, np.float64)
    row_st = np.asarray([r["station"] for r in keys]); row_lead = np.asarray([int(r["lead_minutes"]) for r in keys])
    by_group_day = {}
    for st in STATIONS:
        for lead in LEADS:
            valid = (row_st == st) & (row_lead == lead)
            sse0 = np.bincount(day_idx[valid], weights=base_mse[valid], minlength=len(days_unique))
            sse1 = np.bincount(day_idx[valid], weights=new_mse[valid], minlength=len(days_unique))
            cnt = np.bincount(day_idx[valid], minlength=len(days_unique)).astype(np.float64)
            by_group_day[(st,lead)] = (sse0,sse1,cnt)
    for b in range(args.bootstrap):
        sample = rng.integers(0,len(days_unique),size=len(days_unique)); vals=[]
        for st in STATIONS:
            for lead in LEADS:
                s0,s1,cnt=by_group_day[(st,lead)]
                den=float(cnt[sample].sum())
                vals.append(np.sqrt(float(s1[sample].sum())/den)-np.sqrt(float(s0[sample].sum())/den))
        draws[b]=float(np.mean(vals))
    # The actual paired point metric mirrors the equal-station, equal-lead macro RMSE.
    base_group, new_group = {}, {}
    for st in STATIONS:
        for lead in LEADS:
            mask = np.asarray([r["station"]==st and int(r["lead_minutes"])==lead for r in keys])
            base_group[(st,lead)] = np.sqrt(base_mse[mask])
            new_group[(st,lead)] = np.sqrt(new_mse[mask])
    base_macro = float(np.mean([np.sqrt(np.mean(base_mse[(row_st==st)&(row_lead==ld)])) for st in STATIONS for ld in LEADS]))
    new_macro = float(np.mean([np.sqrt(np.mean(new_mse[(row_st==st)&(row_lead==ld)])) for st in STATIONS for ld in LEADS]))
    station_means = {}
    for st in STATIONS:
        mask = row_st==st
        station_means[st] = {"ACREL_patch_rmse_log1p": float(np.sqrt(np.mean(base_mse[mask]))), "ACRELSP_patch_rmse_log1p": float(np.sqrt(np.mean(new_mse[mask])))}
    out = {"state":"COMPLETE_CROSSFIT_TRAIN_ONLY_SPATIAL_CONTRAST_AUDIT", "split_id":fm["split_id"],
        "method":"future log1p-COT only: preserve ACREL-calibrated center; multiply each of eight neighbor anomalies by train-only gamma[station,lead]",
        "training_reference":"frozen-R COT retrieved from observed training AGRI; reference is not CPP or independent physical truth",
        "gamma_clip":[0.0,2.0], "fit":"through-origin least squares on centered 3x3 log1p-COT neighbor anomalies",
        "train_rows":len(keys), "target_bjt_date_blocks":len(days_unique), "equal_station_equal_lead_patch_rmse_log1p":{"ACREL":base_macro,"ACRELSP":new_macro,"delta_ACRELSP_minus_ACREL":new_macro-base_macro},
        "date_block_bootstrap_equal_station_equal_lead_delta_rmse_ci95":np.quantile(draws,[.025,.975]).tolist(), "station_patch_rmse":station_means,
        "parameters_full_train":full_gamma,"parameters_excluding_each_fold":excluded_gamma,"metrics_by_station_lead_fold":metrics,
        "forecast_pack_manifest_sha256":sha(args.forecast_pack/"pack_manifest.json"),"actual_r_pack_manifest_sha256":sha(args.actual_r_pack/"pack_manifest.json"),
        "forecast_cot_array_sha256":sha(args.forecast_pack/"train"/"cot_log1p.npy"),"actual_r_rows_sha256":sha(args.actual_r_pack/"rows.csv"),
        "center_calibration_sha256":sha(args.center_calibration),"bank_index_sha256":sha(args.bank_index),"fold_oof_ids_sha256":fold_hashes,
        "cpp_used":False,"validation_labels_used_for_fit":False,"test_payload_opened":False,"buffer_payload_opened":False,"future_observed_agri_used_as_inference_input":False}
    args.output.mkdir(parents=True)
    (args.output/"spatial_contrast_calibration.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (args.output/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    print(json.dumps({"state":out["state"],"rmse":out["equal_station_equal_lead_patch_rmse_log1p"],"ci":out["date_block_bootstrap_equal_station_equal_lead_delta_rmse_ci95"],"station":station_means,"test_payload_opened":False},indent=2))


if __name__ == "__main__": main()
