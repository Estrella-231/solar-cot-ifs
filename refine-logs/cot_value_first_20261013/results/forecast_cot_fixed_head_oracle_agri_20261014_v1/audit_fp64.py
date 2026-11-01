import csv,json
from pathlib import Path
import numpy as np
from datetime import datetime,timedelta
p=Path(__file__).parent
with (p/'paired_predictions.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
y=np.array([float(r['ghi_obs_wm2']) for r in rows],np.float64)
a=np.array([float(r['ghi_same_time_R_COT_wm2']) for r in rows],np.float64)
b=np.array([float(r['ghi_forecast_SimVP_R_COT_oracle_AGRI_wm2']) for r in rows],np.float64)
dates=np.array([(datetime.strptime(r['target_time_utc'],'%Y-%m-%d %H:%M:%S')+timedelta(hours=8)).strftime('%Y-%m-%d') for r in rows]);stations=np.array([r['station'] for r in rows]);leads=np.array([int(r['lead_min']) for r in rows]);D=np.unique(dates)
def rm(x,y): return float(np.sqrt(np.mean((x-y)**2)))
by={}
daily={}
for s in ('sili','zhujia'):
 ix=stations==s; by[s]={'n':int(ix.sum()),'unique_target_times':len(set(r['target_time_utc'] for r in rows if r['station']==s)),'unique_dates':len(set(dates[ix])),'RMSE_same_time_R_COT':rm(a[ix],y[ix]),'RMSE_forecast_COT_oracle_AGRI':rm(b[ix],y[ix]),'delta_RMSE':rm(b[ix],y[ix])-rm(a[ix],y[ix]),'MAE_same_time_R_COT':float(np.mean(np.abs(a[ix]-y[ix]))),'MAE_forecast_COT_oracle_AGRI':float(np.mean(np.abs(b[ix]-y[ix])))}
 daily[s]=[]
 for d in D:
  q=ix&(dates==d);daily[s].append((float(np.sum((a[q]-y[q])**2)),float(np.sum((b[q]-y[q])**2)),int(q.sum())))
rng=np.random.default_rng(20261014);B=5000;W=rng.multinomial(len(D),np.full(len(D),1/len(D)),size=B); db=[]
for w in W:
 ds=[]
 for s in ('sili','zhujia'):
  z=np.asarray(daily[s],dtype=np.float64); ss0=w@z[:,0];ss1=w@z[:,1];nn=w@z[:,2]
  ds.append(np.sqrt(ss1/nn)-np.sqrt(ss0/nn))
 db.append(float(np.mean(ds)))
lead=[]
for l in sorted(set(leads)):
 for s in ('sili','zhujia'):
  q=(leads==l)&(stations==s);lead.append({'station':s,'lead_min':int(l),'n':int(q.sum()),'rmse_same_time_R_COT':rm(a[q],y[q]),'rmse_forecast_COT_oracle_AGRI':rm(b[q],y[q]),'delta_rmse':rm(b[q],y[q])-rm(a[q],y[q])})
with (p/'lead_metrics_fp64.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(lead[0]));w.writeheader();w.writerows(lead)
mean0=float(np.mean([by[s]['RMSE_same_time_R_COT'] for s in by]));mean1=float(np.mean([by[s]['RMSE_forecast_COT_oracle_AGRI'] for s in by]))
summary={'state':'INDEPENDENT_FP64_RECOMPUTE_PASS','scope':'validation-only fixed-head offline diagnostic; actual target AGRI/geometry fixed; COT source swapped from same-time frozen-R to SimVP forecast AGRI passed through frozen R','not_deployable_GHI_forecast_score':True,'test_payload_opened':False,'cpp_mask_used_for_sample_filter':False,'n_forecast_instances':len(rows),'n_unique_target_dates':len(D),'dates':list(D),'station_equal_mean_RMSE_same_time_R_COT':mean0,'station_equal_mean_RMSE_forecast_COT_oracle_AGRI':mean1,'station_equal_delta_RMSE':mean1-mean0,'station_equal_delta_RMSE_date_block_bootstrap_95ci':list(map(float,np.quantile(db,[.025,.975]))),'bootstrap_replicates':B,'bootstrap_unit':'BJT target-date blocks (UTC target_time + 8 h), paired; overlapping forecast instances retained within each date','by_station':by,'positive_delta_means':'forecast COT substitution is worse than same-time R-COT under the same frozen head and oracle target AGRI'}
(p/'fp64_independent_audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))


