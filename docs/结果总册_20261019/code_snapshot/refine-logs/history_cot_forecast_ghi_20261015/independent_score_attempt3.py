from pathlib import Path
import json,hashlib
import numpy as np
from datetime import datetime
root=Path(__file__).resolve().parent
R=root/'results_attempt3'
pack=np.load(root/'results_attempt2/paired_pack.npz')
days_init=np.array([datetime.fromisoformat(v).date().isoformat() for v in pack['val_init'].astype(str)])
uniq=sorted(set(days_init.tolist()))
sel={d for d in uniq if int(hashlib.sha256(d.encode()).hexdigest()[:8],16)%4==0}
selmask=np.isin(days_init,list(sel));score_ix=np.flatnonzero(~selmask)
expected_rows=pack['val_rows'][score_ix]
methods=('A','transport','hybrid');seeds=(42,43,44)
pred={};truth={};rows={};leads={};targets={}
for arm in methods:
 for seed in seeds:
  f=np.load(R/f'{arm}_seed{seed}'/'predictions.npz')
  pred[(arm,seed)]=f['pred_ghi'].astype(np.float64)
  truth[(arm,seed)]=f['true_ghi'].astype(np.float64)
  rows[(arm,seed)]=f['row_ids']
  leads[(arm,seed)]=f['lead_minutes'].astype(int)
  targets[(arm,seed)]=f['target_day'].astype(str)
  assert np.array_equal(rows[(arm,seed)],expected_rows),(arm,seed,'row order')
  assert np.array_equal(truth[(arm,seed)],pack['val_ghi'][score_ix])
  assert np.array_equal(leads[(arm,seed)],(pack['val_lead'][score_ix]+1)*15)
  assert np.array_equal(targets[(arm,seed)],pack['val_dates'][score_ix].astype(str))
for arm in methods[1:]:
 for seed in seeds:
  assert np.array_equal(rows[(arm,seed)],rows[('A',seed)])
  assert np.array_equal(truth[(arm,seed)],truth[('A',seed)])
  assert np.array_equal(leads[(arm,seed)],leads[('A',seed)])
  assert np.array_equal(targets[(arm,seed)],targets[('A',seed)])

def metric(p,t):
 e=np.asarray(p,np.float64)-np.asarray(t,np.float64)
 return {'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e))),'bias':float(np.mean(e))}
score_days=np.unique(targets[('A',42)])
rng=np.random.default_rng(20261015);B=10000;draw=rng.integers(0,len(score_days),(B,len(score_days)))
result={};lead_grid=[0,*range(15,241,15)]
for arm in methods[1:]:
 res={}
 for lead in lead_grid:
  mask=np.ones(len(expected_rows),bool) if lead==0 else leads[('A',42)]==lead
  seed_metrics={}
  for a in ('A',arm):
   seed_metrics[a]=[metric(pred[(a,s)][mask],truth[(a,s)][mask]) for s in seeds]
  rmA=np.array([x['rmse'] for x in seed_metrics['A']]);rmC=np.array([x['rmse'] for x in seed_metrics[arm]])
  dates=targets[('A',42)];mass=np.zeros((len(score_days),3,3),np.float64)
  for di,day in enumerate(score_days):
   ix=mask&(dates==day)
   for si,s in enumerate(seeds):
    ea=pred[('A',s)][ix]-truth[('A',s)][ix];ec=pred[(arm,s)][ix]-truth[(arm,s)][ix]
    mass[di,si]=[ix.sum(),np.dot(ea,ea),np.dot(ec,ec)]
  z=mass[draw].sum(1);valid=(z[:,:,0]>0).all(1);z=z[valid]
  bd=(np.sqrt(z[:,:,2]/z[:,:,0])-np.sqrt(z[:,:,1]/z[:,:,0])).mean(1)
  a=float(rmA.mean());c=float(rmC.mean());maeA=np.mean([x['mae'] for x in seed_metrics['A']]);maeC=np.mean([x['mae'] for x in seed_metrics[arm]])
  biasA=np.mean([x['bias'] for x in seed_metrics['A']]);biasC=np.mean([x['bias'] for x in seed_metrics[arm]])
  res[str(lead)]={'n':int(mask.sum()),'A_rmse':a,'COT_rmse':c,'delta_rmse':c-a,'rmse_reduction_pct':100*(a-c)/a,'date_bootstrap_95ci_delta':np.quantile(bd,[.025,.975]).tolist(),'A_mae':float(maeA),'COT_mae':float(maeC),'A_bias':float(biasA),'COT_bias':float(biasC),'per_seed_rmse':{'A':rmA.tolist(),arm:rmC.tolist()}}
 result[arm]=res
summary={'state':'PASS_INDEPENDENT_FP64_RECOMPUTE','scope':'old split Sili validation pilot; model-selection init dates disjoint from scoring init dates; not new-monthly/test','n_validation_rows_total':int(len(pack['val_rows'])),'selection_init_dates':sorted(sel),'score_init_date_count':int(len(set(days_init)-sel)),'scoring_rows':int(len(expected_rows)),'scoring_target_dates':score_days.tolist(),'seeds':list(seeds),'paired_key_check':'PASS exact row_ids, truth, lead, target_day across all arms/seeds','bootstrap':{'cluster':'target BJT day','draws':B,'seed':20261015,'statistic':'three-seed mean of arm-minus-A seed RMSE'},'results':result}
(root/'independent_score_attempt3.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:{str(lead):{'A':v['A_rmse'],'COT':v['COT_rmse'],'pct':v['rmse_reduction_pct'],'CI':v['date_bootstrap_95ci_delta']} for lead,v in res.items() if lead in ('0','60','120','180','240')} for k,res in result.items()},indent=2))
