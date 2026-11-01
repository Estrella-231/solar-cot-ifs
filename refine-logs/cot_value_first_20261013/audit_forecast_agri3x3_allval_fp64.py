#!/usr/bin/env python3
"""FP64 audit of the all-validation SimVP->R bank; CPP is only a sparse crosswalk."""
import csv, datetime as dt, hashlib, json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
OUT=BASE/'results/forecast_agri3x3_allval_20261014_v1'
NPZ=OUT/'forecast_cot_val_bank.npz'
PROFILE=json.loads((OUT/'profile.json').read_text(encoding='utf-8'))
Z=np.load(NPZ,allow_pickle=False)
required=('forecast_log1p_cot','target_agri_r_log1p_cot','persistence_log1p_cot','cpp_reference_cot','cpp_valid_mask','forecast_agri_station_3x3','target_time_utc','dataset_indices','sequence_ids')
assert all(k in Z.files for k in required)
assert PROFILE['state']=='COMPLETE_ALL_VALIDATION_S_TO_R_COT_AND_AGRI3X3_BANK'
assert PROFILE['split']=='val_only' and PROFILE['test_payloads_read']==0
n=PROFILE['windows_completed']; assert n==3140 and PROFILE['finite_windows']==n
shape=Z['forecast_log1p_cot'].shape; assert shape==(n,2,16,16,16)
assert Z['forecast_agri_station_3x3'].shape==(n,2,16,13,3,3)
assert Z['target_time_utc'].shape==(n,16)
assert np.array_equal(Z['dataset_indices'],np.arange(n,dtype=np.int64))
for k in ('target_agri_r_log1p_cot','persistence_log1p_cot','cpp_reference_cot','cpp_valid_mask'):
    assert Z[k].shape==shape
for k in ('forecast_log1p_cot','target_agri_r_log1p_cot','persistence_log1p_cot','forecast_agri_station_3x3'):
    assert np.isfinite(Z[k]).all(),k
mask=Z['cpp_valid_mask'].astype(bool); cpp=Z['cpp_reference_cot'].astype(np.float64)
assert np.isfinite(cpp[mask]).all()
rawhash=hashlib.sha256(NPZ.read_bytes()).hexdigest()
arrhash=hashlib.sha256(np.ascontiguousarray(Z['forecast_log1p_cot']).tobytes()).hexdigest()
assert arrhash==PROFILE['forecast_output_sha256']
# Per-window leads are exact 15-minute target increments.
times=Z['target_time_utc'].astype(str)
for i in range(n):
    ts=[dt.datetime.strptime(v,'%Y-%m-%d %H:%M:%S') for v in times[i]]
    assert all((b-a)==dt.timedelta(minutes=15) for a,b in zip(ts,ts[1:])),i
# The earlier CPP-supported bank must be an exact keyed subset of this bank.
oldpath=BASE/'results/forecast_agri3x3_bank_20261014_v1/forecast_cot_val_bank.npz'
old=np.load(oldpath,allow_pickle=False); oldidx=old['dataset_indices'].astype(np.int64)
assert len(oldidx)==271 and np.all((oldidx>=0)&(oldidx<n))
assert np.array_equal(Z['sequence_ids'][oldidx],old['sequence_ids'])
assert np.array_equal(times[oldidx],old['target_time_utc'].astype(str))
subset_diff=float(np.max(np.abs(Z['forecast_log1p_cot'][oldidx].astype(np.float64)-old['forecast_log1p_cot'].astype(np.float64))))
subset_agri_diff=float(np.max(np.abs(Z['forecast_agri_station_3x3'][oldidx].astype(np.float64)-old['forecast_agri_station_3x3'].astype(np.float64))))
subset_mask_equal=bool(np.array_equal(Z['cpp_valid_mask'][oldidx],old['cpp_valid_mask']))
# Describe forecast-vs-same-R target disagreement. It is NOT an independent COT accuracy metric.
f=np.expm1(Z['forecast_log1p_cot'].astype(np.float64)); r=np.expm1(Z['target_agri_r_log1p_cot'].astype(np.float64))
scales={'1x1':(slice(8,9),slice(8,9)),'3x3':(slice(7,10),slice(7,10)),'16x16':(slice(0,16),slice(0,16))}
def summarize(x,y):
    d=x-y
    return {'n_values':int(d.size),'RMSE':float(np.sqrt(np.mean(d*d,dtype=np.float64))),'MAE':float(np.mean(np.abs(d),dtype=np.float64)),'bias_forecast_minus_target_R':float(np.mean(d,dtype=np.float64))}
rows=[]
for si,station in enumerate(('sili','zhujia')):
  for season,months in (('DJF',(12,1,2)),('MAM',(3,4,5)),('JJA',(6,7,8)),('SON',(9,10,11))):
    for li in range(16):
      date_month=np.array([dt.datetime.strptime(times[i,li],'%Y-%m-%d %H:%M:%S').month for i in range(n)])
      wi=np.flatnonzero(np.isin(date_month,months))
      if len(wi)==0: continue
      for scale,(rs,cs) in scales.items():
        a=f[wi,si,li,rs,cs]; b=r[wi,si,li,rs,cs]
        m=summarize(a,b)
        rows.append({'station':station,'season':season,'lead_min':15*(li+1),'scale':scale,'windows':len(wi),**m})
with (OUT/'allval_R_output_discrepancy_by_season_lead_fp64.csv').open('w',newline='',encoding='utf-8') as q:
    w=csv.DictWriter(q,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
all_metrics={}
for si,station in enumerate(('sili','zhujia')):
  all_metrics[station]={}
  for scale,(rs,cs) in scales.items(): all_metrics[station][scale]=summarize(f[:,si,:,rs,cs],r[:,si,:,rs,cs])
# CPP common-support consistency is preserved as a separate reference comparison.
cpp_rows=[]
for si,station in enumerate(('sili','zhujia')):
  for li in range(16):
    m=mask[:,si,li]
    d={'station':station,'lead_min':15*(li+1),'cpp_valid_pixels':int(m.sum())}
    for name,arr in (('forecast',f),('persistence',np.expm1(Z['persistence_log1p_cot'].astype(np.float64))),('same_time_target_R',r)):
      e=arr[:,si,li][m]-cpp[:,si,li][m]
      d[name+'_RMSE_vs_CPP']=float(np.sqrt(np.mean(e*e,dtype=np.float64))) if e.size else None
    cpp_rows.append(d)
with (OUT/'allval_sparse_CPP_metrics_fp64.csv').open('w',newline='',encoding='utf-8') as q:
    w=csv.DictWriter(q,fieldnames=list(cpp_rows[0]));w.writeheader();w.writerows(cpp_rows)
# Coverage in target BJT calendar; no metrics are inferred from missing CPP.
all_dates=sorted({(dt.datetime.strptime(t,'%Y-%m-%d %H:%M:%S')+dt.timedelta(hours=8)).date().isoformat() for row in times for t in row})
month_counts={}
for day in all_dates:
    month_counts[day[:7]]=month_counts.get(day[:7],0)+1
report={'state':'PASS_INDEPENDENT_FP64_ALLVAL_BANK_AUDIT','split':'validation_only','test_payloads_read':0,'windows':n,'station_order':['sili','zhujia'],'forecast_shape':list(shape),'forecast_agri_3x3_shape':list(Z['forecast_agri_station_3x3'].shape),'unique_sequence_ids':int(len(np.unique(Z['sequence_ids']))),'BJT_target_dates':len(all_dates),'BJT_target_months':month_counts,'CPP_sparse_valid_pixels':int(mask.sum()),'bank_sha256':rawhash,'forecast_array_sha256':arrhash,'profile_array_sha256':PROFILE['forecast_output_sha256'],'overlap_with_prior_CPP_bank':{'windows':len(oldidx),'forecast_COT_max_abs_log1p_difference':subset_diff,'forecast_AGRI_max_abs_difference':subset_agri_diff,'CPP_mask_exact_match':subset_mask_equal},'forecast_vs_same_frozen_R_on_observed_future_AGRI_output_discrepancy_all_windows':all_metrics,'interpretation':'The all-window forecast-vs-target-R comparison measures how SimVP forecast AGRI changes the output of the same frozen retrieval model. Same-time target AGRI->R is an offline retrieval-output reference, not independent COT truth. These overlapping-window discrepancies are descriptive, not deployable COT accuracy. Sparse CPP scores are reported separately on CPP-valid pixels only.','test_closed':True}
(OUT/'allval_bank_fp64_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
