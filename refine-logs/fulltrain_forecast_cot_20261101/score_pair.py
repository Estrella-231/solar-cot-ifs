import json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
B=Path(__file__).resolve().parent
def main():
 runs=[]
 for st in ('sili','zhujia'):
  for seed in (42,43,44):
   d=B/'heads'/f'{st}_seed{seed}'/st/f'seed{seed}'
   a=np.load(d/'A/validation_predictions.npz',allow_pickle=False)
   c=np.load(d/'AC/validation_predictions.npz',allow_pickle=False)
   assert np.array_equal(a['keys'],c['keys']) and np.array_equal(a['truth_ghi_wm2'],c['truth_ghi_wm2'])
   keys=[json.loads(k) for k in a['keys']]
   dates=np.array([(datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date().isoformat() for k in keys])
   leads=np.array([int(k['lead_minutes']) for k in keys])
   err=np.stack([(a['prediction_ghi_wm2']-a['truth_ghi_wm2'])**2,(c['prediction_ghi_wm2']-c['truth_ghi_wm2'])**2])
   runs.append((st,seed,dates,leads,err))
 report={'test_payloads_read':0,'scope':'full-train forecast-input retraining, development validation','comparisons':{}}
 rng=np.random.default_rng(20261101)
 for scope in ('all','sili','zhujia','lead45_240'):
  rs=[r for r in runs if scope not in ('sili','zhujia') or r[0]==scope]
  days=sorted(set(np.concatenate([r[2] for r in rs])))
  blocks=[];point=[]
  for st,seed,dt,ld,e in rs:
   keep=ld>=45 if scope=='lead45_240' else np.ones(len(ld),bool)
   point.append(np.sqrt(e[:,keep].mean(axis=1)))
   sums=np.array([e[:,keep&(dt==day)].sum(axis=1) for day in days]).T
   counts=np.array([sum(keep&(dt==day)) for day in days])
   blocks.append((sums,counts))
  point=np.mean(point,axis=0);boot=[]
  for i in range(5000):
   w=rng.multinomial(len(days),np.full(len(days),1/len(days)))
   x=[np.sqrt((s@w)/(n@w)) for s,n in blocks if n@w>0]
   v=np.mean(x,axis=0);boot.append(v[1]-v[0])
  report['comparisons'][scope]={'no_cot_rmse':float(point[0]),'cot_rmse':float(point[1]),'reduction_percent':float(100*(point[0]-point[1])/point[0]),'delta_ci95':np.quantile(boot,[.025,.975]).tolist(),'target_bjt_dates':len(days)}
 report['lead_metrics']={str(lead):np.mean([np.sqrt(e[:,ld==lead].mean(axis=1)) for _,_,_,ld,e in runs],axis=0).tolist() for lead in range(15,241,15)}
 (B/'RESULTS.json').write_text(json.dumps(report,indent=2))
 (B/'RESULTS.md').write_text('# 全train预测COT配对验证\n\n'+ '\n'.join(f"- {k}: {v['no_cot_rmse']:.3f} → {v['cot_rmse']:.3f} W/m²，下降{v['reduction_percent']:.3f}%，差值95%区间{v['delta_ci95']}" for k,v in report['comparisons'].items())+'\n\n开发验证；未使用test。\n')
if __name__=='__main__':main()
