#!/usr/bin/env python3
"""Descriptive exact-key validation diagnosis of forecast-COT error vs GHI gain."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

STATIONS=("sili","zhujia");SEEDS=(42,43,44)
ARMS=("A","AH","AC","ACREL","ACRELTEMP")
LEAD_BANDS=((15,60,"+15_to_+60"),(75,120,"+75_to_+120"),(135,240,"+135_to_+240"))
COT_BINS=((float("-inf"),np.log1p(1.),"<1"),(np.log1p(1.),np.log1p(10.),"1_to_10"),(np.log1p(10.),np.log1p(30.),"10_to_30"),(np.log1p(30.),float("inf"),">=30"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load_pred(p):
 with np.load(p,allow_pickle=False) as z:
  pred=np.asarray(z["prediction_ghi_wm2"],np.float64);truth=np.asarray(z["truth_ghi_wm2"],np.float64);keys=[json.loads(str(x)) for x in z["keys"]]
 canon=[(k["station"],k["init_time_utc"],k["target_time_utc"],int(k["lead_minutes"]),int(k["ghi_row_index"])) for k in keys]
 if pred.shape!=truth.shape or pred.ndim!=1 or len(canon)!=len(pred) or len(set(canon))!=len(canon) or not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError(f"invalid prediction file: {p}")
 return canon,pred,truth
def main():
 p=argparse.ArgumentParser();p.add_argument("--forecast-pack",type=Path,required=True);p.add_argument("--actual-r-pack",type=Path,required=True);p.add_argument("--calibration",type=Path,required=True);p.add_argument("--baseline-root",type=Path,required=True);p.add_argument("--temporal-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=2000);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 fm=json.loads((a.forecast_pack/"pack_manifest.json").read_text());rm=json.loads((a.actual_r_pack/"pack_manifest.json").read_text());cal=json.loads(a.calibration.read_text())
 if fm.get("state")!="COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK" or fm.get("train_forecast_oof") is not True or fm.get("forecast_bank_oof_fold_count")!=5:raise ValueError("causal five-fold OOF forecast pack required")
 if fm.get("test_payload_opened") is not False or fm.get("buffer_payload_opened") is not False or fm.get("future_observed_agri_used_as_input") is not False or fm.get("cpp_used_as_input") is not False:raise ValueError("forecast pack closure failed")
 if rm.get("state")!="COMPLETE" or rm.get("split_id")!=fm.get("split_id") or rm.get("test_payload_opened") is not False:raise ValueError("actual-R split/provenance mismatch")
 if cal.get("state")!="COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION" or cal.get("forecast_pack_manifest_sha256")!=sha(a.forecast_pack/"pack_manifest.json"):raise ValueError("future-COT calibration provenance mismatch")
 if cal.get("test_payload_opened") is not False or cal.get("buffer_payload_opened") is not False:raise ValueError("calibration test/buffer closure failed")
 keys=list(csv.DictReader((a.forecast_pack/"val"/"keys.csv").open(newline="")));n=len(keys)
 if n!=fm.get("rows",{}).get("val") or n!=66590:raise ValueError(f"unexpected validation key count {n}")
 cots=np.load(a.forecast_pack/"val"/"cot_log1p.npy",mmap_mode="r",allow_pickle=False)
 if cots.shape!=(n,3,1,3,3) or cots.dtype!=np.float32:raise ValueError("forecast COT array shape/dtype mismatch")
 rrows=list(csv.DictReader((a.actual_r_pack/"rows.csv").open(newline="")));ridx={}
 for i,r in enumerate(rrows):
  if r["split"]=="val":
   k=(r["station"],r["target_time_utc"])
   if k in ridx:raise ValueError("duplicate validation actual-R timestamp")
   ridx[k]=i
 rcot=np.load(a.actual_r_pack/"cot_log1p.npy",mmap_mode="r",allow_pickle=False)
 if rcot.shape!=(len(rrows),1,3,3) or rcot.dtype!=np.float32:raise ValueError("actual-R COT shape/dtype mismatch")
 rows=[];missing=0
 for i,k in enumerate(keys):
  st=k["station"];ld=int(k["lead_minutes"]);ridx0=ridx.get((st,k["target_time_utc"]))
  if ridx0 is None:missing+=1;continue
  raw=np.asarray(cots[i,2,0],np.float32);coef=cal["parameters_full_train"][st][str(ld)];pred=coef["intercept"]+coef["slope"]*raw
  truth=np.asarray(rcot[ridx0,0],np.float32);t=datetime.strptime(k["target_time_utc"],"%Y-%m-%d %H:%M:%S");day=(t+timedelta(hours=8)).strftime("%Y-%m-%d")
  center=float(pred[1,1]);bin_name=next(label for lo,hi,label in COT_BINS if lo<=center<hi)
  rows.append({"idx":i,"station":st,"lead":ld,"lead_band":next(label for lo,hi,label in LEAD_BANDS if lo<=ld<=hi),"target_bjt_date":day,"pred_center_log1p":center,"cot_regime":bin_name,"center_cot_error_log1p":float(center-truth[1,1]),"center_cot_abs_error_log1p":float(abs(center-truth[1,1])),"center_cot_sq_error_log1p":float((center-truth[1,1])**2),"patch_cot_sq_error_log1p":float(np.mean((pred-truth)**2))})
 if len(rows)<60000:raise ValueError(f"low exact target COT coverage: {len(rows)}/{n}")
 # Exact saved GHI predictions and labels across all methods/seeds.
 preds={};keys_by_station={};truth_by_station={}
 for seed in SEEDS:
  for st in STATIONS:
   station_refkeys=None;station_truth=None
   for arm in ("A","AH","AC","ACREL"):
    path=a.baseline_root/st/f"seed{seed}"/arm/"validation_predictions.npz";kk,pp,yy=load_pred(path)
    if station_refkeys is None:station_refkeys=kk;station_truth=yy
    if kk!=station_refkeys or not np.array_equal(yy,station_truth):raise ValueError("baseline prediction keys/truth differ")
    preds[(seed,st,arm)]=pp
   path=a.temporal_root/st/f"seed{seed}"/"ACRELTEMP"/"validation_predictions.npz";kk,pp,yy=load_pred(path)
   if kk!=station_refkeys or not np.array_equal(yy,station_truth):raise ValueError("temporal prediction keys/truth differ")
   preds[(seed,st,"ACRELTEMP")]=pp
   expected=[(k["station"],k["init_time_utc"],k["target_time_utc"],int(k["lead_minutes"]),int(k["ghi_row_index"])) for k in keys if k["station"]==st]
   if station_refkeys!=expected:raise ValueError(f"forecast COT and GHI rows are not exact-key aligned: {st}")
   if seed==SEEDS[0]:keys_by_station[st]=station_refkeys;truth_by_station[st]=station_truth
   elif station_refkeys!=keys_by_station[st] or not np.array_equal(station_truth,truth_by_station[st]):raise ValueError(f"sample coverage varies across seeds: {st}")
 err={}
 for seed in SEEDS:
  for st in STATIONS:
   for arm in ARMS:err[(seed,st,arm)]=preds[(seed,st,arm)]-truth_by_station[st]
 key_to_row={st:{k:i for i,k in enumerate(keys_by_station[st])} for st in STATIONS}
 for r in rows:
  k=keys[r["idx"]];kk=(k["station"],k["init_time_utc"],k["target_time_utc"],int(k["lead_minutes"]),int(k["ghi_row_index"]));r["gidx"]=key_to_row[r["station"]][kk];i=r["gidx"]
  r["acrel_minus_ah_delta_sse_mean_seed"]=float(np.mean([err[(s,r["station"],"ACREL")][i]**2-err[(s,r["station"],"AH")][i]**2 for s in SEEDS]))
  r["acrel_minus_a_delta_sse_mean_seed"]=float(np.mean([err[(s,r["station"],"ACREL")][i]**2-err[(s,r["station"],"A")][i]**2 for s in SEEDS]))
 days=sorted({r["target_bjt_date"] for r in rows});dindex={d:i for i,d in enumerate(days)};rng=np.random.default_rng(20261020);outrows=[]
 groups=[]
 for st in STATIONS:
  for blo,bhi,band in LEAD_BANDS:
   groups.append((st,band,"all"))
   for _,_,regime in COT_BINS:groups.append((st,band,regime))
 for st,band,regime in groups:
  ix=np.asarray([j for j,r in enumerate(rows) if r["station"]==st and r["lead_band"]==band and (regime=="all" or r["cot_regime"]==regime)],np.int64)
  if len(ix)<100:continue
  rs=[rows[j] for j in ix];loc=np.asarray([r["gidx"] for r in rs],np.int64);group_days=sorted({r["target_bjt_date"] for r in rs});gd={d:i for i,d in enumerate(group_days)};di=np.asarray([gd[r["target_bjt_date"]] for r in rs],np.int32);cnt=np.bincount(di,minlength=len(group_days)).astype(float)
  x=np.asarray([r["center_cot_sq_error_log1p"] for r in rs],np.float64);ae=np.asarray([r["center_cot_abs_error_log1p"] for r in rs],np.float64)
  def rmse_delta(seed,base,sel=None):
   if sel is None:sel=np.ones(len(loc),bool)
   ids=loc[sel];
   return float(np.sqrt(np.mean(err[(seed,st,"ACREL")][ids]**2))-np.sqrt(np.mean(err[(seed,st,base)][ids]**2)))
  m_acrel_ah=float(np.mean([rmse_delta(s,"AH") for s in SEEDS]));m_acrel_a=float(np.mean([rmse_delta(s,"A") for s in SEEDS]))
  # Date-block bootstrap within this descriptive subgroup; fixed global target-date blocks.
  draws1=np.empty(a.bootstrap);draws2=np.empty(a.bootstrap)
  se=[err[(s,st,"ACREL")] for s in SEEDS];sh=[err[(s,st,"AH")] for s in SEEDS];sa=[err[(s,st,"A")] for s in SEEDS]
  sse1=np.asarray([[np.bincount(di,weights=v[loc]**2,minlength=len(group_days)) for v in (se[z],sh[z])] for z in range(3)])
  sse2=np.asarray([[np.bincount(di,weights=v[loc]**2,minlength=len(group_days)) for v in (se[z],sa[z])] for z in range(3)])
  for b in range(a.bootstrap):
   mult=np.bincount(rng.integers(0,len(group_days),size=len(group_days)),minlength=len(group_days)).astype(float);dd1=[];dd2=[]
   for z in range(3):
    den=float(np.sum(cnt*mult));dd1.append(np.sqrt(float(np.sum(sse1[z,0]*mult))/den)-np.sqrt(float(np.sum(sse1[z,1]*mult))/den));dd2.append(np.sqrt(float(np.sum(sse2[z,0]*mult))/den)-np.sqrt(float(np.sum(sse2[z,1]*mult))/den))
   draws1[b]=np.mean(dd1);draws2[b]=np.mean(dd2)
  outrows.append({"station":st,"lead_band":band,"predicted_cot_regime":regime,"n":len(ix),"target_bjt_days":len(group_days),"center_cot_rmse_log1p":float(np.sqrt(np.mean(x))),"center_cot_mae_log1p":float(np.mean(ae)),"center_cot_bias_log1p":float(np.mean([r["center_cot_error_log1p"] for r in rs])),"patch_cot_rmse_log1p":float(np.sqrt(np.mean([r["patch_cot_sq_error_log1p"] for r in rs]))),"ACREL_minus_AH_GHI_rmse_wm2":m_acrel_ah,"ACREL_minus_AH_ci95_dateblock_wm2":np.quantile(draws1,[.025,.975]).tolist(),"ACREL_minus_A_GHI_rmse_wm2":m_acrel_a,"ACREL_minus_A_ci95_dateblock_wm2":np.quantile(draws2,[.025,.975]).tolist(),"mean_ACREL_minus_AH_delta_sse_wm4":float(np.mean([r["acrel_minus_ah_delta_sse_mean_seed"] for r in rs]))})
 # Descriptive sample-level COT/GHI relationship over exact common rows.
 ce=np.asarray([r["center_cot_abs_error_log1p"] for r in rows]);dg=np.asarray([r["acrel_minus_ah_delta_sse_mean_seed"] for r in rows]);corr=float(np.corrcoef(ce,dg)[0,1]);
 result={"state":"COMPLETE_VALIDATION_DIAGNOSTIC_COT_ERROR_VS_GHI_INCREMENT","question":"association of exact-time forecast-COT center error against frozen-R reference with paired GHI benefit","valid_rows_total":n,"valid_rows_exact_actual_R":len(rows),"rows_without_exact_actual_R":missing,"target_bjt_date_blocks":len(days),"seeds":list(SEEDS),"stations":list(STATIONS),"lead_bands":[{"label":x[2],"min":x[0],"max":x[1]} for x in LEAD_BANDS],"forecast_cot_regimes":[x[2] for x in COT_BINS],"sample_level_pearson_abs_cot_error_vs_acrel_minus_ah_delta_sse":corr,"group_metrics":outrows,"keys_sha256":sha(a.forecast_pack/"val"/"keys.csv"),"forecast_cot_sha256":sha(a.forecast_pack/"val"/"cot_log1p.npy"),"actual_r_manifest_sha256":sha(a.actual_r_pack/"pack_manifest.json"),"calibration_sha256":sha(a.calibration),"cpp_used":False,"actual_r_used_only_as_validation_diagnostic_reference":True,"future_observed_agri_used_as_inference_input":False,"test_payload_opened":False,"buffer_payload_opened":False,"validation_used_for_fitting_or_threshold_selection":False,"interpretation_limit":"overlapping forecast instances are clustered by target BJT date; R is a retrieval comparator, not independent physical truth"}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 with a.output.with_suffix(".csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
 print(json.dumps({k:result[k] for k in ("state","valid_rows_total","valid_rows_exact_actual_R","rows_without_exact_actual_R","target_bjt_date_blocks","sample_level_pearson_abs_cot_error_vs_acrel_minus_ah_delta_sse","group_metrics","test_payload_opened")},indent=2))
if __name__=="__main__":main()
