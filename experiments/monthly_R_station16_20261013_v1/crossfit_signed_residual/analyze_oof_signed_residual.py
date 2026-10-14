#!/usr/bin/env python3
"""Independent validation recomputation for the cross-fitted residual run."""
from __future__ import annotations
import argparse, csv, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)
ARMS = ("A", "AGRI", "AGRI_COT")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bjt_day(ts):
    return (datetime.fromisoformat(ts.replace("Z", "+00:00")) + timedelta(hours=8)).date().isoformat()


def metrics(pred, truth):
    e = np.asarray(pred, np.float64) - np.asarray(truth, np.float64)
    return {"rmse_wm2": float(np.sqrt(np.mean(e*e))),
            "mae_wm2": float(np.mean(np.abs(e))),
            "bias_wm2": float(np.mean(e))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--oof-root", type=Path, required=True)
    p.add_argument("--residual-root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--pbs-job-id", default="not_recorded")
    a = p.parse_args()
    oof_status = json.loads((a.oof_root/"oof_status.json").read_text())
    residual_status = json.loads((a.residual_root/"experiment_status.json").read_text())
    pack_meta = json.loads((a.pack/"pack_manifest.json").read_text())
    if oof_status.get("state") != "COMPLETE" or residual_status.get("state") != "COMPLETE":
        raise ValueError("experiment completion gate failed")
    if oof_status.get("test_payload_opened") is not False or residual_status.get("test_payload_opened") is not False or pack_meta.get("test_payload_opened") is not False:
        raise ValueError("test-closed provenance gate failed")
    if oof_status.get("pack_manifest_sha256") != residual_status.get("pack_manifest_sha256"):
        raise ValueError("OOF and residual pack manifests differ")
    rows = read_rows(a.pack/"rows.csv")
    key_to_i = {(r["station"], r["target_time_utc"]): i for i, r in enumerate(rows)}
    cot = np.load(a.pack/"cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    runs, per_seed, subgroups, a_gap = {}, [], [], []
    for station in STATIONS:
        for seed in SEEDS:
            keys = [r for r in rows if r["station"] == station and r["split"] == "val"]
            ids = np.asarray([key_to_i[(station, r["target_time_utc"])] for r in keys], np.int64)
            truth = np.asarray([float(r["ghi"]) for r in keys], np.float64)
            dates = np.asarray([bjt_day(r["target_time_utc"]) for r in keys])
            d = a.residual_root/f"seed{seed}"/station
            baseline = a.baseline_root/f"ghi_heads_new_r_seed{seed}_v1"/station
            pred = {"A": np.load(d/"A_full_val_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)}
            deltas = {}
            for arm in ("AGRI", "AGRI_COT"):
                pred[arm] = np.load(d/f"{arm}_val_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)
                deltas[arm] = np.load(d/f"{arm}_val_signed_residual_wm2.npy", allow_pickle=False).astype(np.float64)
            if any(len(v) != len(keys) for v in [*pred.values(), *deltas.values()]):
                raise ValueError(f"row count mismatch {station}/{seed}")
            saved_old = np.load(baseline/"A_val_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)
            reload_delta = float(np.max(np.abs(saved_old-pred["A"])))
            if reload_delta > 1.0:
                raise ValueError(f"full A reload mismatch {station}/{seed}: {reload_delta}")
            scores = {arm: metrics(pred[arm], truth) for arm in ARMS}
            meta_by_arm = {arm: json.loads((d/f"{arm}_validation.json").read_text()) for arm in ("AGRI", "AGRI_COT")}
            if meta_by_arm["AGRI"]["trainable_parameters"] != meta_by_arm["AGRI_COT"]["trainable_parameters"]:
                raise ValueError("residual model parameter count mismatch")
            run = {"station": station, "seed": seed, "keys": keys, "dates": dates,
                   "truth": truth, "pred": pred, "deltas": deltas, "scores": scores,
                   "cot": np.expm1(np.asarray(cot[ids, 0, 1, 1], np.float64)),
                   "clear": np.asarray([float(r["clear"]) for r in keys]),
                   "kt": np.asarray([float(r["kt"]) for r in keys]),
                   "reload_delta": reload_delta, "metadata": meta_by_arm}
            runs[(station, seed)] = run
            train_keys = [r for r in rows if r["station"] == station and r["split"] == "train"]
            train_i = np.asarray([key_to_i[(station, r["target_time_utc"])] for r in train_keys], np.int64)
            oof_dir = a.oof_root/f"seed{seed}"/station
            oof_rows = np.load(oof_dir/"train_pack_row_indices.npy", allow_pickle=False)
            oof_pred = np.load(oof_dir/"A_oof_train_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)
            fold_ids = np.load(oof_dir/"oof_fold_id.npy", allow_pickle=False)
            full_train = np.load(d/"A_full_train_prediction_ghi_wm2.npy", allow_pickle=False).astype(np.float64)
            if not np.array_equal(oof_rows, train_i) or any(len(v) != len(train_i) for v in (oof_pred, fold_ids, full_train)):
                raise ValueError(f"OOF/full-A training alignment mismatch {station}/{seed}")
            train_truth = np.asarray([float(r["ghi"]) for r in train_keys], np.float64)
            for fold in sorted(np.unique(fold_ids).tolist()):
                m = fold_ids == fold
                e_oof = oof_pred[m] - train_truth[m]
                e_full = full_train[m] - train_truth[m]
                a_gap.append({"station": station, "seed": seed, "fold": int(fold),
                    "n": int(m.sum()), "oof_rmse_wm2": float(np.sqrt(np.mean(e_oof**2))),
                    "full_A_rmse_same_rows_wm2": float(np.sqrt(np.mean(e_full**2))),
                    "oof_bias_wm2": float(np.mean(e_oof)), "full_A_bias_same_rows_wm2": float(np.mean(e_full)),
                    "oof_error_std_wm2": float(np.std(e_oof)), "full_A_error_std_same_rows_wm2": float(np.std(e_full)),
                    "paired_prediction_rmse_gap_wm2": float(np.sqrt(np.mean((oof_pred[m]-full_train[m])**2)))})
            rec = {"station": station, "seed": seed, "n": len(keys),
                   "A_reload_max_abs_delta_wm2": reload_delta,
                   "best_epoch_AGRI": meta_by_arm["AGRI"]["best_epoch"],
                   "best_epoch_AGRI_COT": meta_by_arm["AGRI_COT"]["best_epoch"],
                   "positive_delta_fraction_AGRI": float(np.mean(deltas["AGRI"]>0)),
                   "positive_delta_fraction_AGRI_COT": float(np.mean(deltas["AGRI_COT"]>0))}
            for arm in ARMS:
                for k,v in scores[arm].items(): rec[f"{arm}_{k}"] = v
            rec["AGRI_minus_A_rmse"] = scores["AGRI"]["rmse_wm2"]-scores["A"]["rmse_wm2"]
            rec["AGRI_COT_minus_A_rmse"] = scores["AGRI_COT"]["rmse_wm2"]-scores["A"]["rmse_wm2"]
            rec["AGRI_COT_minus_AGRI_rmse"] = scores["AGRI_COT"]["rmse_wm2"]-scores["AGRI"]["rmse_wm2"]
            per_seed.append(rec)
            under = truth > pred["A"]
            high_thick_bright = (run["cot"] >= 3.0) & (run["clear"] >= 100.0) & (run["kt"] >= 0.9)
            over = truth < pred["A"]
            for subset, mask in (("A_underpredict", under), ("A_overpredict", over),
                                 ("bright_thick_COT_ge3_clear_ge100_kt_ge0p9", high_thick_bright)):
                if not mask.any(): continue
                for arm in ("AGRI", "AGRI_COT"):
                    delta = deltas[arm][mask]
                    subgroups.append({"station": station, "seed": seed, "subset": subset,
                        "arm": arm, "n": int(mask.sum()),
                        "positive_adjustment_fraction": float(np.mean(delta>0)),
                        "right_sign_fraction": float(np.mean(delta*(truth[mask]-pred["A"][mask])>0)),
                        "absolute_error_improved_fraction": float(np.mean(np.abs(pred[arm][mask]-truth[mask]) < np.abs(pred["A"][mask]-truth[mask]))),
                        "A_rmse_wm2": metrics(pred["A"][mask],truth[mask])["rmse_wm2"],
                        "candidate_rmse_wm2": metrics(pred[arm][mask],truth[mask])["rmse_wm2"]})
    date_union = sorted({d for run in runs.values() for d in run["dates"].tolist()})
    date_idx = {d:i for i,d in enumerate(date_union)}
    rng = np.random.default_rng(a.seed)
    draws = rng.integers(0, len(date_union), size=(a.bootstrap,len(date_union)))
    uncertainty = {}
    for left,right,name in (("AGRI","A","AGRI_minus_A"),
                            ("AGRI_COT","A","AGRI_COT_minus_A"),
                            ("AGRI_COT","AGRI","AGRI_COT_minus_AGRI")):
        station_boot = {}
        point_station = {}
        for station in STATIONS:
            sse_by_arm = {}
            n_by_arm = None
            for arm in (left,right):
                rows_sse=[]; rows_n=[]
                for seed in SEEDS:
                    run=runs[(station,seed)]
                    di=np.asarray([date_idx[d] for d in run["dates"]],np.int64)
                    nday=np.bincount(di,minlength=len(date_union)).astype(np.float64)
                    sse=np.bincount(di,weights=(run["pred"][arm]-run["truth"])**2,minlength=len(date_union))
                    rows_sse.append(sse); rows_n.append(nday)
                sse_by_arm[arm]=np.stack(rows_sse)
                n_by_arm=np.stack(rows_n)
            sampled_n=n_by_arm[:,draws].sum(axis=2)
            lrm=np.sqrt(sse_by_arm[left][:,draws].sum(axis=2)/sampled_n).mean(axis=0)
            rrm=np.sqrt(sse_by_arm[right][:,draws].sum(axis=2)/sampled_n).mean(axis=0)
            station_boot[station]=lrm-rrm
            point_station[station]=float(np.mean([runs[(station,s)]["scores"][left]["rmse_wm2"]-runs[(station,s)]["scores"][right]["rmse_wm2"] for s in SEEDS]))
        allboot=np.mean(np.stack([station_boot[s] for s in STATIONS]),axis=0)
        uncertainty[name]={"left_minus_right_equal_station_rmse_wm2":float(np.mean(list(point_station.values()))),
            "by_station_wm2":point_station,
            "date_block_bootstrap_95pct_ci_wm2":[float(x) for x in np.quantile(allboot,[.025,.975])],
            "replicates":a.bootstrap,"bootstrap_unit":"common BJT validation target-date blocks; seeds averaged within station",
            "interpretation":"validation-only; conditional on validation-selected checkpoints"}
    station_summary={}
    for arm in ARMS:
        vals={s:float(np.mean([runs[(s,seed)]["scores"][arm]["rmse_wm2"] for seed in SEEDS])) for s in STATIONS}
        station_summary[arm]={"rmse_by_station_mean_over_seeds":vals,"equal_station_mean_rmse_wm2":float(np.mean(list(vals.values())))}
    oof_summary=[]
    for r in oof_status["runs"]:
        oof_summary.append({"station":r["station"],"seed":r["seed"],"n_train_rows":r["n_train_rows"],
                            "oof_train_rmse_wm2":r["oof_rmse_wm2"],"unique_train_dates":r["unique_train_dates"],
                            "fold_rmse_wm2":[x["heldout_rmse_wm2"] for x in r["folds"]]})
    a.output.mkdir(parents=True,exist_ok=True)
    for name, records in (("per_seed_metrics.csv",per_seed),("direction_subgroups.csv",subgroups),
                          ("oof_full_A_train_fold_gap.csv", a_gap)):
        with (a.output/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    ordered_runs=[runs[(s,sd)] for s in STATIONS for sd in SEEDS]
    np.savez_compressed(a.output/"validation_predictions_and_keys.npz",
        station=np.concatenate([np.repeat(r["keys"][0]["station"],len(r["keys"])) for r in ordered_runs]).astype("U16"),
        seed=np.concatenate([np.repeat(np.int16(sd),len(r["keys"])) for r in ordered_runs for sd in [r["metadata"]["AGRI"]["seed"]]]),
        target_time_utc=np.concatenate([[k["target_time_utc"] for k in r["keys"]] for r in ordered_runs]).astype("U32"),
        bjt_date=np.concatenate([r["dates"] for r in ordered_runs]).astype("U16"),
        ghi_truth_wm2=np.concatenate([r["truth"] for r in ordered_runs]).astype(np.float32),
        cot_center=np.concatenate([r["cot"] for r in ordered_runs]).astype(np.float32),
        clear_sky_wm2=np.concatenate([r["clear"] for r in ordered_runs]).astype(np.float32),
        observed_kt=np.concatenate([r["kt"] for r in ordered_runs]).astype(np.float32),
        pred_A_wm2=np.concatenate([r["pred"]["A"] for r in ordered_runs]).astype(np.float32),
        pred_AGRI_wm2=np.concatenate([r["pred"]["AGRI"] for r in ordered_runs]).astype(np.float32),
        pred_AGRI_COT_wm2=np.concatenate([r["pred"]["AGRI_COT"] for r in ordered_runs]).astype(np.float32),
        delta_AGRI_wm2=np.concatenate([r["deltas"]["AGRI"] for r in ordered_runs]).astype(np.float32),
        delta_AGRI_COT_wm2=np.concatenate([r["deltas"]["AGRI_COT"] for r in ordered_runs]).astype(np.float32),
        test_payload_opened=np.asarray(False))
    summary={"state":"INDEPENDENT_RECOMPUTATION_COMPLETE","split_id":pack_meta["split_id"],
        "test_payload_opened":False,"training_jobs":{"pbs_job_id":a.pbs_job_id,"state":"COMPLETE"},
        "pack_manifest_sha256":oof_status["pack_manifest_sha256"],"n_station_seed_pairs":len(runs),
        "validation_dates_bjt":len(date_union),"validation_metrics":station_summary,
        "paired_date_block_uncertainty":uncertainty,"oof_A_training_diagnostics":oof_summary,
        "oof_design":"5 contiguous date-block folds, one adjacent available train date buffered on each side, fixed 30 epochs, no validation selection",
        "interpretation_boundary":"same-time real AGRI to GHI retrieval; not SimVP forecast-chain performance"}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
