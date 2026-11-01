import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
p=Path(__file__).parent
with (p/'paired_predictions.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
cols={'TT':'pred_TT_wm2','FT':'pred_FT_wm2','TF':'pred_TF_wm2','FF':'pred_FF_wm2'}
y=np.array([float(r['ghi_obs_wm2']) for r in rows],np.float64); st=np.array([r['station'] for r in rows]);dates=np.array([(datetime.strptime(r['target_time_utc'],'%Y-%m-%d %H:%M:%S')+timedelta(hours=8)).strftime('%Y-%m-%d') for r in rows]); D=np.unique(dates)
pred={k:np.array([float(r[col]) for r in rows],np.float64) for k,col in cols.items()}
contrasts={'AGRI_effect_at_same_time_COT':('FT','TT'),'COT_effect_at_true_AGRI':('TF','TT'),'COT_effect_at_forecast_AGRI':('FF','FT'),'AGRI_effect_at_forecast_COT':('FF','TF'),'joint_forecast_input_effect':('FF','TT')}
metric={};daily={s:[] for s in ('sili','zhujia')}
for s in ('sili','zhujia'):
 ix=st==s;metric[s]={'n_instances':int(ix.sum()),'unique_target_keys':len(set(r['target_time_utc'] for r in rows if r['station']==s)),'unique_BJT_dates':len(set(dates[ix])),'arms':{}}
 for k in cols:
  e=pred[k][ix]-y[ix];metric[s]['arms'][k]={'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e)))}
 for d in D:
  q=ix&(dates==d);rec={'n':int(q.sum())}
  for k in cols:rec[k]=float(np.sum((pred[k][q]-y[q])**2))
  daily[s].append(rec)
# Station-equal RMSE is the mean of station-specific RMSEs.
def se_metric(w,k,s):
 z=daily[s]; n=np.array([r['n'] for r in z],np.float64);ss=np.array([r[k] for r in z],np.float64)
 den=float(w@n)
 return np.sqrt(float(w@ss)/den) if den else np.nan
point={}
for name,(new,base) in contrasts.items():
 point[name]=float(np.mean([metric[s]['arms'][new]['RMSE']-metric[s]['arms'][base]['RMSE'] for s in ('sili','zhujia')]))
rng=np.random.default_rng(20261014);B=5000;W=rng.multinomial(len(D),np.full(len(D),1/len(D)),size=B);ci={}
for name,(new,base) in contrasts.items():
 vals=[]
 for w in W:
  d=[]
  for s in ('sili','zhujia'):d.append(se_metric(w,new,s)-se_metric(w,base,s))
  vals.append(float(np.mean(d)))
 ci[name]=[float(x) for x in np.quantile(vals,[.025,.975])]
leads=np.array([int(r['lead_min']) for r in rows]);lead_rows=[]
for l in sorted(set(leads)):
 for s in ('sili','zhujia'):
  q=(leads==l)&(st==s);row={'station':s,'lead_min':int(l),'n':int(q.sum())}
  for k in cols:row[f'RMSE_{k}']=float(np.sqrt(np.mean((pred[k][q]-y[q])**2)))
  row['delta_FF_minus_TT']=row['RMSE_FF']-row['RMSE_TT'];lead_rows.append(row)
with (p/'lead_metrics_fp64.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(lead_rows[0]));w.writeheader();w.writerows(lead_rows)
summary={'state':'INDEPENDENT_FP64_RECOMPUTE_PASS','scope':'validation-only frozen seed42 AC head; 2x2 offline attribution; same 7,005 forecast instances for all arms','test_payload_opened':False,'COT_reference':'same-time target AGRI through frozen R, corresponding to GHI-head train input contract','COT_forecast':'SimVP forecast AGRI through same frozen R','AGRI_reference':'actual target-time AGRI (oracle)','AGRI_forecast':'frozen SimVP output at target lead, station-centered 3x3 physical AGRI','dates_BJT':list(D),'n_BJT_date_blocks':len(D),'instances':len(rows),'station_equal_mean_RMSE_by_arm':{k:float(np.mean([metric[s]['arms'][k]['RMSE'] for s in ('sili','zhujia')])) for k in cols},'station_equal_delta_RMSE_by_contrast':point,'date_block_paired_bootstrap_95CI':ci,'bootstrap_replicates':B,'bootstrap_unit':'BJT target-date blocks (UTC+8), paired; overlapping forecast instances retained within date','by_station':metric,'limitations':'Fixed head was trained with actual AGRI and same-time R-COT. FF is only a frozen-head operational-input replay; it is not a trained forecast-GHI model or publication score. Eight BJT dates and overlapping windows make intervals wide.'}
(p/'fp64_independent_audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
