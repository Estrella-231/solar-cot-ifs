#!/usr/bin/env python3
"""Independent date-block paired audit for attempt14 COT validation predictions."""
from __future__ import annotations
import argparse, csv, datetime as dt, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

LEADS=tuple(range(15,241,15))
SITES={"Niushou":(8,6),"Wangzhai":(9,8),"Mizhuang":(7,10)}
LOG_DENOM=math.log(101.0)
STRATA=("near_zero_lt1","thin_1_10","moderate_10_30","thick_ge30")

def csvrows(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def load_pred(path):
    z=np.load(path,allow_pickle=False)
    times=[str(x) for x in z["init_time_bjt"].tolist()]
    x=np.asarray(z["pred_log_scaled"],dtype=np.float64)
    if x.shape != (len(times),16,1,16,16) or len(set(times))!=len(times): raise ValueError(f"bad prediction schema {path}: {x.shape}")
    return {t:x[i] for i,t in enumerate(times)}
def pack_stats(): return [0.0,0.0,0.0,0]
def add(s,e): s[0]+=e*e; s[1]+=abs(e); s[2]+=e; s[3]+=1
def score(stats, method, draw, stratum=None):
    cells=sorted({(k[1],k[2]) for k in stats if k[0]==method and ((stratum is None and len(k)==4) or (stratum is not None and len(k)==5 and k[4]==stratum))})
    vals=[]
    for cell in cells:
        ss,sa,se,n=0.0,0.0,0.0,0
        for date in draw:
            key=(method,cell[0],cell[1],date) if stratum is None else (method,cell[0],cell[1],date,stratum)
            v=stats.get(key)
            if v is not None: ss+=v[0]; sa+=v[1]; se+=v[2]; n+=v[3]
        if n: vals.append((math.sqrt(ss/n),sa/n,se/n))
    if not vals: return [float("nan")]*3
    return [float(np.mean([v[0] for v in vals])),float(np.mean([v[1] for v in vals])),float(np.mean([v[2] for v in vals]))]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--attempt13-pred",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--bootstrap",type=int,default=5000); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    methods={"log42":a.attempt13_pred,"hybrid42":a.run_root/"hybrid_seed42/validation_forecast_predictions.npz","log43":a.run_root/"log_only_seed43/validation_forecast_predictions.npz","hybrid43":a.run_root/"hybrid_seed43/validation_forecast_predictions.npz","log44":a.run_root/"log_only_seed44/validation_forecast_predictions.npz","hybrid44":a.run_root/"hybrid_seed44/validation_forecast_predictions.npz"}
    pred={k:load_pred(v) for k,v in methods.items()}
    common=set.intersection(*(set(x) for x in pred.values()))
    if any(set(x)!=common for x in pred.values()): raise ValueError("model prediction init keys differ; do not silently score unequal cohorts")
    report=json.load(open(a.cache/"cache_report.json",encoding="utf-8"))
    if report.get("test_payloads_read",0)!=0 or report.get("buffer_payloads_read",0)!=0: raise ValueError("cache audit indicates test/buffer access")
    rows=csvrows(a.cache/"bank_rows.csv"); bank={dt.datetime.strptime(r["time_bjt"],"%Y%m%d%H%M"):int(r["index"]) for r in rows}; split={dt.datetime.strptime(r["time_bjt"],"%Y%m%d%H%M"):r["split"].lower() for r in rows}
    pairs=[r for r in csvrows(a.cache/"forecast_pairs.csv") if r["split"].lower()=="val"]
    arr={n:np.load(a.cache/f"{n}.npy",mmap_mode="r",allow_pickle=False) for n in ("history_cot_log","history_valid","cpp_cot","cpp_valid")}
    stats=defaultdict(pack_stats); date_set=set(); pair_set=set(); pred_oob={k:0 for k in [*pred,"persistence"]}; confusion={k:[0,0,0] for k in [*pred,"persistence"]};
    for row in pairs:
        init_s=row["init_time_bjt"]; ti=dt.datetime.strptime(init_s,"%Y%m%d%H%M"); tt=dt.datetime.strptime(row["target_time_bjt"],"%Y%m%d%H%M"); lead=int(row["lead_minutes"])
        if init_s not in common or tt-ti!=dt.timedelta(minutes=lead) or lead not in LEADS: raise ValueError(f"invalid/unmatched validation key {row}")
        if split.get(ti)!="val" or split.get(tt)!="val": raise ValueError("cross-split pair")
        ii,yi=bank[ti],bank[tt]; k=lead//15-1; date=tt.strftime("%Y%m%d"); date_set.add(date)
        h=np.expm1(np.asarray(arr["history_cot_log"][ii,0],dtype=np.float64)*LOG_DENOM); hv=np.asarray(arr["history_valid"][ii,0],dtype=bool); y=np.asarray(arr["cpp_cot"][yi,0],dtype=np.float64); yv=np.asarray(arr["cpp_valid"][yi,0],dtype=bool)
        forecasts={m:np.expm1(v[init_s][k,0]*LOG_DENOM) for m,v in pred.items()}
        for m,x in forecasts.items(): pred_oob[m]+=int(np.count_nonzero((x<0)|(x>100)))
        for site,(rr,cc) in SITES.items():
            if not(yv[rr,cc] and hv[rr,cc]): continue
            pair_set.add((init_s,lead,row["target_time_bjt"]))
            truth=float(y[rr,cc]); truth_stratum="near_zero_lt1" if truth<1 else "thin_1_10" if truth<10 else "moderate_10_30" if truth<30 else "thick_ge30"
            estimates={**forecasts,"persistence":float(h[rr,cc])}
            for m,x in estimates.items():
                estimate=float(x[rr,cc] if isinstance(x,np.ndarray) else x)
                e=estimate-truth
                add(stats[(m,site,lead,date)],e); add(stats[(m,site,lead,date,truth_stratum)],e)
                if estimate>=30 and truth>=30: confusion[m][0]+=1
                elif estimate>=30 and truth<30: confusion[m][1]+=1
                elif truth>=30 and estimate<30: confusion[m][2]+=1
    dates=sorted(date_set); full_draw=dates
    point={m:score(stats,m,full_draw) for m in [*pred,"persistence"]}
    stratum_rmse={m:{s:score(stats,m,full_draw,s)[0] for s in STRATA} for m in [*pred,"persistence"]}
    # Verify that the independent scorer reproduces the already saved per-arm scorer.
    scored={}
    for m,path in methods.items():
        score_path=path.parent/"validation_score.json"
        if m=="log42": score_path=path.parent/"validation_score.json"
        saved=json.load(open(score_path,encoding="utf-8"))
        if saved.get("test_payloads_read")!=0 or saved.get("buffer_payloads_read")!=0: raise ValueError(f"test/buffer not closed in {m}")
        want=saved["equal_station_equal_lead"]["forecast_rmse_cot"]
        if abs(point[m][0]-want)>1e-8: raise ValueError(f"independent FP64 RMSE mismatch {m}: {point[m][0]} vs {want}")
        scored[m]=saved
    pairs_by_seed={42:("hybrid42","log42"),43:("hybrid43","log43"),44:("hybrid44","log44")}
    rng=np.random.default_rng(a.seed); boot_pair=np.empty(a.bootstrap); boot_persist=np.empty(a.bootstrap)
    for b in range(a.bootstrap):
        draw=list(rng.choice(dates,size=len(dates),replace=True))
        boot_pair[b]=float(np.mean([score(stats,h,draw)[0]-score(stats,l,draw)[0] for h,l in pairs_by_seed.values()]))
        boot_persist[b]=float(np.mean([score(stats,h,draw)[0]-score(stats,"persistence",draw)[0] for h,_ in pairs_by_seed.values()]))
    delta_seed={str(seed):point[h][0]-point[l][0] for seed,(h,l) in pairs_by_seed.items()}
    avg_hybrid=float(np.mean([point[h][0] for h,_ in pairs_by_seed.values()])); avg_log=float(np.mean([point[l][0] for _,l in pairs_by_seed.values()])); avg_persistence=point["persistence"][0]
    thick_delta={str(seed):stratum_rmse[h]["thick_ge30"]-stratum_rmse[l]["thick_ge30"] for seed,(h,l) in pairs_by_seed.items()}
    thick_avg=float(np.mean([stratum_rmse[h]["thick_ge30"] for h,_ in pairs_by_seed.values()])); log_thick_avg=float(np.mean([stratum_rmse[l]["thick_ge30"] for _,l in pairs_by_seed.values()])); pers_thick=stratum_rmse["persistence"]["thick_ge30"]
    det={}
    for m,(tp,fp,fn) in confusion.items(): det[m]={"tp":tp,"fp":fp,"fn":fn,"pod":tp/(tp+fn) if tp+fn else None,"far":fp/(tp+fp) if tp+fp else None,"csi":tp/(tp+fp+fn) if tp+fp+fn else None}
    metrics={m:{"rmse_cot":v[0],"mae_cot":v[1],"bias_cot":v[2],"thick_ge30_rmse_cot":stratum_rmse[m]["thick_ge30"],"thick_ge30_bias_cot":score(stats,m,full_draw,"thick_ge30")[2],"thick_detection":det[m],"out_of_range_pred_pixel_count":pred_oob[m]} for m,v in point.items()}
    pair_ci=np.quantile(boot_pair,[.025,.975]).tolist(); pers_ci=np.quantile(boot_persist,[.025,.975]).tolist()
    result={"state":"ATTEMPT14_INDEPENDENT_FP64_PAIRED_DATE_BLOCK_AUDIT","scope":"Hubei station-centered pixel, validation only, CPP retrieval reference, not independent truth","test_payloads_read":0,"buffer_payloads_read":0,"init_lead_pairs":len(pair_set),"station_pixel_records":sum(v[3] for key,v in stats.items() if key[0]=="persistence" and len(key)==4),"target_dates_bjt":dates,"prediction_keys_common":len(common),"bootstrap_replicates":a.bootstrap,"bootstrap_seed":a.seed,"per_model_metrics_equal_station_equal_lead":metrics,"seed_paired_hybrid_minus_log_only_rmse":delta_seed,"seed_mean_rmse":{"hybrid":avg_hybrid,"log_only":avg_log,"persistence":avg_persistence},"seed_mean_hybrid_minus_log_only_rmse":float(np.mean(list(delta_seed.values()))),"seed_mean_hybrid_minus_log_only_date_block_95pct_ci":pair_ci,"seed_mean_hybrid_minus_persistence_rmse":avg_hybrid-avg_persistence,"seed_mean_hybrid_minus_persistence_date_block_95pct_ci":pers_ci,"thick_ge30_rmse":{"hybrid_seed_mean":thick_avg,"log_only_seed_mean":log_thick_avg,"persistence":pers_thick,"seed_paired_differences_hybrid_minus_log_only":thick_delta},"passes_all_seed_overall_point_gate":all(v<0 for v in delta_seed.values()),"passes_date_block_pair_gate":pair_ci[1]<0,"passes_persistence_date_block_gate":pers_ci[1]<0,"passes_thick_cloud_gate":thick_avg<=log_thick_avg and thick_avg<=pers_thick,"exploratory_validation_only":True}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    # Paired seed-level date-block scores for transparent audit/replotting.
    with a.output.with_name("paired_seed_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["seed","hybrid_rmse_cot","log_only_rmse_cot","delta_hybrid_minus_logonly","hybrid_thick_rmse","log_only_thick_rmse","delta_thick"])
        for seed,(h,l) in pairs_by_seed.items(): w.writerow([seed,point[h][0],point[l][0],delta_seed[str(seed)],stratum_rmse[h]["thick_ge30"],stratum_rmse[l]["thick_ge30"],thick_delta[str(seed)]])
    print(json.dumps({k:result[k] for k in ("state","init_lead_pairs","station_pixel_records","seed_mean_rmse","seed_mean_hybrid_minus_log_only_rmse","seed_mean_hybrid_minus_log_only_date_block_95pct_ci","thick_ge30_rmse","passes_all_seed_overall_point_gate","passes_date_block_pair_gate","passes_persistence_date_block_gate","passes_thick_cloud_gate")},ensure_ascii=False))
if __name__=="__main__": main()
