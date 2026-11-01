#!/usr/bin/env python3
"""Independent FP64 audit of the selected seed-42 classifier gate on validation."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLP = np.load(ROOT / "clp_trainval.npz", allow_pickle=False)
RP = np.load(ROOT / "R_validation_predictions_seed42.npz", allow_pickle=False)
CP = np.load(ROOT / "remote_result_retry1" / "validation_predictions.npz", allow_pickle=False)
if not np.array_equal(RP["sample_indices"], CP["sample_indices"]):
    raise ValueError("frozen R and classifier validation sample indices differ")
idx = RP["sample_indices"].astype(np.int64)
if not np.array_equal(CLP["sample_indices"][idx], idx):
    raise ValueError("CLP sidecar does not align to R validation indices")
if not np.array_equal(CLP["station"][idx], CP["station"]):
    raise ValueError("station keys mismatch")
if not np.array_equal(CLP["target_time_utc"][idx], CP["target_time_utc"]):
    raise ValueError("target time keys mismatch")

mask = RP["mask"][:, 0].astype(bool)
if not np.array_equal(mask, CP["valid"].astype(bool)):
    raise ValueError("R and classifier validation masks differ")
clp = CLP["clp"][idx, 0]
if np.any(mask & ~np.isin(clp, (0, 1, 2))):
    raise ValueError("missing CLP under the common validation mask")
cloud = clp >= 1
prob = CP["prob_cloud"].astype(np.float64)
pred_gate_cloud = prob >= 0.5
raw_cot = np.maximum(np.expm1(RP["pred_log1p_cot"][:, 0].astype(np.float64)), 0.0)
ref_cot = RP["reference_cot"][:, 0].astype(np.float64)
gate_cot = raw_cot * pred_gate_cloud
raw_cloud_proxy = raw_cot >= 0.1
raw_thick, ref_thick, gate_thick = raw_cot >= 30.0, ref_cot >= 30.0, gate_cot >= 30.0
stations = CP["station"].astype(str)
times = CP["target_time_utc"].astype(str)
dates = np.asarray([(datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d") for t in times])


def class_scores(pred, target, valid):
    tp = int(np.count_nonzero(pred & target & valid)); fp = int(np.count_nonzero(pred & ~target & valid))
    fn = int(np.count_nonzero(~pred & target & valid)); tn = int(np.count_nonzero(~pred & ~target & valid))
    return {"n": int(valid.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "POD": tp/(tp+fn) if tp+fn else None,
            "FAR": fp/(tp+fp) if tp+fp else None,
            "CSI": tp/(tp+fp+fn) if tp+fp+fn else None}


def score(sel):
    out = {}
    for s in ("sili", "zhujia"):
        sm = sel & (stations == s)
        v = mask[sm]
        pr, tr, pg = raw_cot[sm], ref_cot[sm], gate_cot[sm]
        cl, pc, rp = cloud[sm], pred_gate_cloud[sm], raw_cloud_proxy[sm]
        t, prt, pgt = ref_thick[sm], raw_thick[sm], gate_thick[sm]
        out[s] = {
            "cloud_classifier": class_scores(pc, cl, v),
            "R_COT_ge0.1_proxy": class_scores(rp, cl, v),
            "COT_RMSE_raw": float(np.sqrt(np.mean((pr[v]-tr[v])**2))),
            "COT_RMSE_gated": float(np.sqrt(np.mean((pg[v]-tr[v])**2))),
            "thick_raw": class_scores(prt, t, v), "thick_gated": class_scores(pgt, t, v),
            "valid_pixels": int(v.sum()),
        }
    out["station_equal_mean"] = {
        "COT_RMSE_raw": float(np.mean([out[s]["COT_RMSE_raw"] for s in ("sili", "zhujia")])),
        "COT_RMSE_gated": float(np.mean([out[s]["COT_RMSE_gated"] for s in ("sili", "zhujia")])),
        "cloud_FAR_raw_proxy": float(np.mean([out[s]["R_COT_ge0.1_proxy"]["FAR"] for s in ("sili", "zhujia")])),
        "cloud_FAR_gate": float(np.mean([out[s]["cloud_classifier"]["FAR"] for s in ("sili", "zhujia")])),
    }
    return out


full = score(np.ones(len(stations), bool))
pooled_cot_rmse = {
    "raw": float(np.sqrt(np.mean((raw_cot[mask]-ref_cot[mask])**2))),
    "gated": float(np.sqrt(np.mean((gate_cot[mask]-ref_cot[mask])**2))),
}
unique_dates = np.unique(dates)
# Date-cluster bootstrap from per-station, per-date sufficient statistics.
per = {}
for s in ("sili", "zhujia"):
    per[s] = []
    for d in unique_dates:
        sel = (stations == s) & (dates == d)
        v = mask[sel]
        if not v.any():
            per[s].append(None); continue
        rr, gg, yy = raw_cot[sel], gate_cot[sel], ref_cot[sel]
        c, cg, rp = raw_cloud_proxy[sel], pred_gate_cloud[sel], cloud[sel]
        tt, rt, gt = ref_thick[sel], raw_thick[sel], gate_thick[sel]
        per[s].append({
            "n": int(v.sum()), "sse_raw": float(np.sum((rr[v]-yy[v])**2)), "sse_gate": float(np.sum((gg[v]-yy[v])**2)),
            "cloud_tp_raw": int(np.count_nonzero(c & rp & v)), "cloud_fp_raw": int(np.count_nonzero(c & ~rp & v)),
            "cloud_tp_gate": int(np.count_nonzero(cg & rp & v)), "cloud_fp_gate": int(np.count_nonzero(cg & ~rp & v)),
            "thick_tp_raw": int(np.count_nonzero(rt & tt & v)), "thick_fp_raw": int(np.count_nonzero(rt & ~tt & v)), "thick_fn_raw": int(np.count_nonzero(~rt & tt & v)),
            "thick_tp_gate": int(np.count_nonzero(gt & tt & v)), "thick_fp_gate": int(np.count_nonzero(gt & ~tt & v)), "thick_fn_gate": int(np.count_nonzero(~gt & tt & v)),
            "cloud_tp_ref": int(np.count_nonzero(rp & v)),
        })

rng = np.random.default_rng(20261014)
boots = {"delta_station_equal_COT_RMSE": [], "delta_station_equal_cloud_FAR": [],
         "sili_delta_cloud_POD": [], "zhujia_delta_cloud_POD": [],
         "sili_delta_thick_POD": [], "zhujia_delta_thick_POD": [],
         "sili_delta_thick_CSI": [], "zhujia_delta_thick_CSI": []}
for _ in range(2000):
    draws = rng.integers(0, len(unique_dates), size=len(unique_dates))
    accum = {}
    for s in ("sili", "zhujia"):
        vals = [per[s][i] for i in draws if per[s][i] is not None]
        a = {k: sum(q[k] for q in vals) for k in vals[0]}
        rm0, rm1 = np.sqrt(a["sse_raw"]/a["n"]), np.sqrt(a["sse_gate"]/a["n"])
        far0 = a["cloud_fp_raw"]/(a["cloud_fp_raw"]+a["cloud_tp_raw"])
        far1 = a["cloud_fp_gate"]/(a["cloud_fp_gate"]+a["cloud_tp_gate"])
        pod0 = a["cloud_tp_raw"]/a["cloud_tp_ref"]
        pod1 = a["cloud_tp_gate"]/a["cloud_tp_ref"]
        tp0,fp0,fn0=a["thick_tp_raw"],a["thick_fp_raw"],a["thick_fn_raw"]
        tp1,fp1,fn1=a["thick_tp_gate"],a["thick_fp_gate"],a["thick_fn_gate"]
        accum[s] = dict(drm=rm1-rm0, dfar=far1-far0, dpod=pod1-pod0,
                        dtpod=tp1/(tp1+fn1)-tp0/(tp0+fn0),
                        dtcsi=tp1/(tp1+fp1+fn1)-tp0/(tp0+fp0+fn0))
    boots["delta_station_equal_COT_RMSE"].append((accum["sili"]["drm"]+accum["zhujia"]["drm"])/2)
    boots["delta_station_equal_cloud_FAR"].append((accum["sili"]["dfar"]+accum["zhujia"]["dfar"])/2)
    for s in ("sili", "zhujia"):
        boots[f"{s}_delta_cloud_POD"].append(accum[s]["dpod"])
        boots[f"{s}_delta_thick_POD"].append(accum[s]["dtpod"])
        boots[f"{s}_delta_thick_CSI"].append(accum[s]["dtcsi"])
ci = {k: [float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in boots.items()}
delta_rm = full["station_equal_mean"]["COT_RMSE_gated"] - full["station_equal_mean"]["COT_RMSE_raw"]
delta_far = full["station_equal_mean"]["cloud_FAR_gate"] - full["station_equal_mean"]["cloud_FAR_raw_proxy"]
result = {
    "state": "PASS_INDEPENDENT_FP64_CLOUD_GATE_AUDIT", "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
    "selected_checkpoint_epoch": 24, "probability_threshold": 0.5, "cloud_proxy_threshold_cot": 0.1,
    "thick_threshold_cot": 30.0, "samples": int(len(stations)), "BJT_target_dates": int(len(unique_dates)),
    "valid_pixels": int(mask.sum()), "common_R_CLP_mask_exact": True, "test_payloads_read": 0,
    "by_station": {s: full[s] for s in ("sili", "zhujia")},
    "station_equal_mean": full["station_equal_mean"],
    "pixel_pooled_COT_RMSE": pooled_cot_rmse,
    "paired_date_cluster_bootstrap_95ci": ci, "bootstrap_replicates": 2000,
    "deltas_gate_minus_raw_proxy": {"station_equal_COT_RMSE": delta_rm, "station_equal_cloud_FAR": delta_far},
    "pre_registered_B1_gate": {
        "COT_RMSE_reduced": bool(delta_rm < 0), "cloud_FAR_reduced": bool(delta_far < 0),
        "cloud_FAR_reduced_each_station": all(full[s]["cloud_classifier"]["FAR"] < full[s]["R_COT_ge0.1_proxy"]["FAR"] for s in ("sili","zhujia")),
        "cloud_POD_drop_le_1pp_each_station": all(full[s]["cloud_classifier"]["POD"]-full[s]["R_COT_ge0.1_proxy"]["POD"] >= -0.01 for s in ("sili","zhujia")),
        "thick_POD_drop_le_1pp_each_station": all(full[s]["thick_gated"]["POD"]-full[s]["thick_raw"]["POD"] >= -0.01 for s in ("sili","zhujia")),
        "thick_CSI_drop_le_0.01_each_station": all(full[s]["thick_gated"]["CSI"]-full[s]["thick_raw"]["CSI"] >= -0.01 for s in ("sili","zhujia")),
        "proceed_to_B2": False,
        "reason": "Gate threshold is fixed at 0.5; station-equal COT RMSE does not decrease and Sili cloud POD drops by more than 1 percentage point. Stop this candidate per preregistered rule; do not tune threshold or launch GHI retraining."
    },
    "provenance": {"classifier_output": "remote_result_retry1/validation_predictions.npz",
                   "R_output": "R_validation_predictions_seed42.npz", "CLP_sidecar": "clp_trainval.npz"}
}
(ROOT/"independent_audit_fp64.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
