#!/usr/bin/env python3
"""Exact-key, date-block paired validation audit for spatial-contrast COT."""
from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS=("sili","zhujia")
SEEDS=(42,43,44)
BASE_ARMS=("A","AH","AC","ACREL")

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load(p:Path):
    m=json.loads((p.parent/"metrics.json").read_text(encoding="utf-8")); c=p.parent/"best.pt"
    if m.get("predictions_sha256")!=sha(p) or m.get("checkpoint_sha256")!=sha(c): raise ValueError(f"hash mismatch: {p}")
    with np.load(p,allow_pickle=False) as z:
        pred=np.asarray(z["prediction_ghi_wm2"],np.float64); truth=np.asarray(z["truth_ghi_wm2"],np.float64)
        keys=[json.loads(str(x)) for x in z["keys"]]
    canon=[(k["station"],k["init_time_utc"],k["target_time_utc"],int(k["lead_minutes"]),int(k["ghi_row_index"])) for k in keys]
    if pred.shape!=truth.shape or pred.ndim!=1 or len(canon)!=len(pred) or len(set(canon))!=len(canon): raise ValueError(f"invalid predictions: {p}")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all(): raise ValueError(f"nonfinite predictions: {p}")
    days=np.asarray([(datetime.strptime(k[2],"%Y-%m-%d %H:%M:%S")+timedelta(hours=8)).strftime("%Y-%m-%d") for k in canon])
    return {"pred":pred,"truth":truth,"keys":canon,"days":days,"leads":np.asarray([k[3] for k in canon],np.int16)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--candidate-root",type=Path,required=True);p.add_argument("--baseline-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--candidate-arm",default="ACRELSP");p.add_argument("--bootstrap",type=int,default=2000);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    cm=[]
    for seed in SEEDS:
        m=json.loads((a.candidate_root/f"seed{seed}_manifest.json").read_text(encoding="utf-8"))
        if m.get("state")!="COMPLETE_FUTURE_GHI_COT_ABLATION_SEED_RUN" or m.get("arm")!=a.candidate_arm or m.get("test_payload_opened") is not False or m.get("buffer_payload_opened") is not False: raise ValueError(f"candidate seed {seed} incomplete or invalid")
        cm.append(m)
    data={}
    for seed in SEEDS:
        for st in STATIONS:
            base={arm:load(a.baseline_root/st/f"seed{seed}"/arm/"validation_predictions.npz") for arm in BASE_ARMS}
            cand=load(a.candidate_root/st/f"seed{seed}"/a.candidate_arm/"validation_predictions.npz")
            ref=base["A"]
            for arm,item in [*base.items(),(a.candidate_arm,cand)]:
                if item["keys"]!=ref["keys"] or not np.array_equal(item["truth"],ref["truth"]): raise ValueError(f"key/truth mismatch {st}/{seed}/{arm}")
            if any(k[0]!=st for k in ref["keys"]): raise ValueError("station key mismatch")
            data[(seed,st)]={**base,a.candidate_arm:cand}
    for st in STATIONS:
        ref=data[(SEEDS[0],st)]["A"]
        for seed in SEEDS[1:]:
            got=data[(seed,st)]["A"]
            if got["keys"]!=ref["keys"] or not np.array_equal(got["truth"],ref["truth"]): raise ValueError("sample coverage varies across seeds")
    days=sorted(set().union(*(set(data[(s,st)]["A"]["days"]) for s in SEEDS for st in STATIONS)))
    di={d:i for i,d in enumerate(days)}; arms=(*BASE_ARMS,a.candidate_arm)
    rmse_by={}; lead_rows=[]
    for st in STATIONS:
        rmse_by[st]={}
        for arm in arms:
            vals=[]
            for seed in SEEDS:
                e=data[(seed,st)][arm]["pred"]-data[(seed,st)][arm]["truth"]
                vals.append(float(np.sqrt(np.mean(e*e))))
                leads=data[(seed,st)][arm]["leads"]
                for lead in sorted(set(leads.tolist())):
                    ix=leads==lead; ee=e[ix]
                    lead_rows.append({"station":st,"seed":seed,"arm":arm,"lead_minutes":int(lead),"n":int(ix.sum()),"rmse_wm2":float(np.sqrt(np.mean(ee*ee))),"mae_wm2":float(np.mean(np.abs(ee))),"bias_wm2":float(np.mean(ee))})
            rmse_by[st][arm]=float(np.mean(vals))
    point={arm:float(np.mean([rmse_by[st][arm] for st in STATIONS])) for arm in arms}
    comparisons=tuple((a.candidate_arm,r) for r in ("AH","A","ACREL","AC"))
    rng=np.random.default_rng(20261020);draws={f"{l}_minus_{r}":np.empty(a.bootstrap) for l,r in comparisons}
    # Store per-date squared errors and counts, preserving paired rows and station/seed structure.
    bd={}
    for seed in SEEDS:
        for st in STATIONS:
            base=data[(seed,st)]["A"]; ids=np.asarray([di[d] for d in base["days"]],np.int32)
            for arm in arms:
                err=data[(seed,st)][arm]["pred"]-data[(seed,st)][arm]["truth"]
                bd[(seed,st,arm)]=(np.bincount(ids,weights=err*err,minlength=len(days)),np.bincount(ids,minlength=len(days)).astype(float))
    for b in range(a.bootstrap):
        sample=rng.integers(0,len(days),size=len(days));count=np.bincount(sample,minlength=len(days)).astype(float); vals={k:[] for k in draws}
        for seed in SEEDS:
            for st in STATIONS:
                for arm in arms:
                    sse,n=bd[(seed,st,arm)]; den=float(np.sum(n*count)); vals0=np.sqrt(float(np.sum(sse*count))/den)
                    for left,right in comparisons:
                        if arm==left: vals[f"{left}_minus_{right}"].append(vals0)
                        elif arm==right: vals[f"{left}_minus_{right}"].append(-vals0)
        for key in draws: draws[key][b]=float(np.mean(vals[key]))
    ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()}
    lead_summary=[]
    for st in STATIONS:
        for lo,hi,label in ((15,60,"+15_to_+60"),(75,120,"+75_to_+120"),(135,240,"+135_to_+240")):
            vals={arm:[] for arm in arms}
            for seed in SEEDS:
                for arm in arms:
                    x=data[(seed,st)][arm]; e=x["pred"]-x["truth"]; ix=(x["leads"]>=lo)&(x["leads"]<=hi)
                    vals[arm].append(float(np.sqrt(np.mean(e[ix]**2))))
            lead_summary.append({"station":st,"lead_band":label,**{arm:float(np.mean(v)) for arm,v in vals.items()},f"{a.candidate_arm}_minus_AH":float(np.mean(vals[a.candidate_arm])-np.mean(vals["AH"])),f"{a.candidate_arm}_minus_ACREL":float(np.mean(vals[a.candidate_arm])-np.mean(vals["ACREL"]))})
    result={"state":"COMPLETE_PAIRED_FUTURE_GHI_COT_ABLATION_VALIDATION_AUDIT","candidate_arm":a.candidate_arm,"screening":"five-fold OOF train forecasts; validation only; independent test not accessed","seeds":list(SEEDS),"stations":list(STATIONS),"target_bjt_date_blocks":len(days),"bootstrap_replicates":a.bootstrap,"bootstrap_seed":20261020,"metric":f"equal-station mean of seed-averaged station RMSE; negative delta favors {a.candidate_arm}","equal_station_mean_seed_rmse_wm2":point,"pairwise_delta_rmse_wm2":{f"{l}_minus_{r}":point[l]-point[r] for l,r in comparisons},"date_block_bootstrap_95pct_ci_wm2":ci,"per_station_mean_seed_rmse_wm2":rmse_by,"lead_band_metrics":lead_summary,"per_seed_station_lead_metrics":lead_rows,"paired_key_rule":"exact station/init/target/lead/GHI-row keys and truth arrays required across baseline/candidate/seeds","candidate_manifests_sha256":{str(s):sha(a.candidate_root/f"seed{s}_manifest.json") for s in SEEDS},"test_payload_opened":False,"buffer_payload_opened":False,"future_observed_agri_used_as_inference_input":False,"cpp_used_as_input":False,"claim_rule":"validation screening only; training uses five-fold OOF forecast inputs; no independent-test claim"}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with a.output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
        import csv
        w=csv.DictWriter(f,fieldnames=list(lead_rows[0]));w.writeheader();w.writerows(lead_rows)
    print(json.dumps({k:result[k] for k in ("state","candidate_arm","equal_station_mean_seed_rmse_wm2","pairwise_delta_rmse_wm2","date_block_bootstrap_95pct_ci_wm2","target_bjt_date_blocks","test_payload_opened")},indent=2))

if __name__=="__main__": main()
