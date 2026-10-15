#!/usr/bin/env python3
"""Independent FP64 common-support scoring and date-cluster paired bootstrap."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def metrics(predlog, ref, mask):
    pred=np.expm1(np.asarray(predlog,np.float64)); y=np.asarray(ref,np.float64); m=np.asarray(mask,bool)
    if not m.any(): return dict(n=0,rmse=None,mae=None,bias=None)
    e=pred[m]-y[m]
    return dict(n=int(m.sum()),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),bias=float(np.mean(e)))


def load_pred(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in ("array_indices","pred_log1p_cot","reference_cot","mask")}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--pack",type=Path,required=True); p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--bootstrap",type=int,default=2000)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    rows=list(csv.DictReader((a.pack/"rows_supervised.csv").open(newline="",encoding="utf-8")))
    valids=np.asarray([i for i,r in enumerate(rows) if r["split"]=="val"],np.int64)
    paths={"original_checkpoint":a.runs/"original_checkpoint_validation_predictions.npz",
           "old_geometry_retrain":a.runs/"old/validation_predictions.npz",
           "corrected_geometry_retrain":a.runs/"corrected/validation_predictions.npz"}
    arrays={k:load_pred(v) for k,v in paths.items()}
    ref=arrays["original_checkpoint"]["reference_cot"].astype(np.float64)
    mask=arrays["original_checkpoint"]["mask"].astype(bool)
    for name,z in arrays.items():
        if not np.array_equal(z["array_indices"],valids): raise ValueError(f"{name}: validation row order mismatch")
        if not np.array_equal(z["reference_cot"],ref,equal_nan=True) or not np.array_equal(z["mask"].astype(bool),mask):
            raise ValueError(f"{name}: not the exact same validation labels/mask")
        if not np.isfinite(z["pred_log1p_cot"][mask]).all(): raise ValueError(f"{name}: nonfinite predictions under mask")
    vrows=[rows[i] for i in valids]
    station=np.asarray([r["station"] for r in vrows]); day=np.asarray([r["bjt"][:8] for r in vrows])
    strata={"nearzero_[0,0.1)":mask&(ref<.1),"thin_[0.1,5)":mask&(ref>=.1)&(ref<5),
            "moderate_[5,30)":mask&(ref>=5)&(ref<30),"thick_[30,100]":mask&(ref>=30)&(ref<=100)}
    pred={k:np.expm1(z["pred_log1p_cot"].astype(np.float64)) for k,z in arrays.items()}
    summary={"all":{},"by_station":{},"by_reference_bin":{},"paired_rmse_deltas":{}}
    for name in arrays: summary["all"][name]=metrics(arrays[name]["pred_log1p_cot"],ref,mask)
    for s in ("sili","zhujia"):
        si=station==s; summary["by_station"][s]={name:metrics(arrays[name]["pred_log1p_cot"][si],ref[si],mask[si]) for name in arrays}
    for label,sm in strata.items():
        summary["by_reference_bin"][label]={name:metrics(arrays[name]["pred_log1p_cot"],ref,sm) for name in arrays}
    pairs=[("corrected_geometry_retrain","old_geometry_retrain"),
           ("corrected_geometry_retrain","original_checkpoint"),
           ("old_geometry_retrain","original_checkpoint")]
    # Equal-station primary score, resampling whole BJT dates while preserving
    # all pixels/rows and method pairing within each selected date.
    row_has_support=mask.reshape(len(mask),-1).any(1)
    unique={s:np.unique(day[(station==s)&row_has_support]) for s in ("sili","zhujia")}
    groups={s:{d:np.where((station==s)&(day==d)&row_has_support)[0] for d in unique[s]} for s in unique}
    def rmse(name,s,selected_days):
        ids=np.concatenate([groups[s][d] for d in selected_days])
        return metrics(arrays[name]["pred_log1p_cot"][ids],ref[ids],mask[ids])["rmse"]
    point={}
    for left,right in pairs:
        per={}
        for s in ("sili","zhujia"):
            per[s]=summary["by_station"][s][left]["rmse"]-summary["by_station"][s][right]["rmse"]
        point[f"{left}_minus_{right}"]=dict(by_station=per,equal_station_mean=float(np.mean(list(per.values()))))
    rng=np.random.default_rng(20261015); draws={k:[] for k in point}
    for _ in range(a.bootstrap):
        sampled={s:rng.choice(unique[s],size=len(unique[s]),replace=True) for s in unique}
        score={}
        for left,right in pairs:
            diffs=[]
            for s in ("sili","zhujia"):
                ids=np.concatenate([groups[s][d] for d in sampled[s]])
                l=metrics(arrays[left]["pred_log1p_cot"][ids],ref[ids],mask[ids])["rmse"]
                r=metrics(arrays[right]["pred_log1p_cot"][ids],ref[ids],mask[ids])["rmse"]
                diffs.append(l-r)
            key=f"{left}_minus_{right}"; score[key]=float(np.mean(diffs)); draws[key].append(score[key])
    for key,val in point.items():
        d=np.asarray(draws[key]); val["date_cluster_bootstrap_95pct"]=[float(np.quantile(d,.025)),float(np.quantile(d,.975))]
        val["bootstrap_replicates"]=a.bootstrap
    report=dict(state="PASS_INDEPENDENT_FP64_PAIRED_R_SCORE",split_id="hunan_monthly_block_2d_buffer1d_seed42_v1",
                common_validation_rows=len(valids),common_validation_pixels=int(mask.sum()),unique_bjt_dates={s:len(unique[s]) for s in unique},
                primary="equal-station mean of station RMSE; paired BJT-date cluster bootstrap",metrics=summary,
                paired_rmse_deltas=point,files={k:sha(v) for k,v in paths.items()},
                pack_audit_sha256=sha(a.pack/"pack_audit.json"),test_payloads_read=0,
                claim_scope="CPP-supported daytime retrieval-reference validation on the old/new geometry common pixel set")
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
