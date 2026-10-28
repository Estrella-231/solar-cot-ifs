"""Array-only fixed-head audit scores. No retraining or scientific model ranking."""
from pathlib import Path
import csv,json
import numpy as np
B=Path(__file__).parent
with (B/'attempt1/ghi_rows.csv').open() as f:rows=list(csv.DictReader(f))
sites=np.array([r['station'] for r in rows]);receipt=json.loads((B/'audit_counterfactuals1/receipt.json').read_text())
assert receipt['status']=='PASS'
with np.load(B/'audit_counterfactuals1/predictions.npz') as z:
 y=z['truth'];result={}
 names=['real_predicted_COT','future_COT_train_mean','all_COT_train_mean']+[f'shuffled_future_COT_{i:02d}' for i in range(20)]
 for name in names:
  per=[]
  for s in ('sili','zhujia'):
   for seed in (42,43,44):
    ids=sites==s;p=z[f'{s}_seed{seed}_{name}'][ids]
    assert np.isfinite(p).all();e=p-y[ids];per.append({'station':s,'seed':seed,'RMSE':float(np.sqrt(np.mean(e*e)))})
  result[name]={'macro_RMSE_Wm2':float(np.mean([v['RMSE'] for v in per])),'per_run':per}
sh=np.array([result[n]['macro_RMSE_Wm2'] for n in names[3:]])
with np.load(B/'audit_features1/features.npz') as base,np.load(B/'audit_canary1/features.npz') as can:
 check={k:{'equal':bool(np.array_equal(base[k],can[k])),'max_abs_delta':float(abs(base[k]-can[k]).max())} for k in base.files}
 assert check['dydiff']['equal'] and check['simvp']['equal'] and not check['observed']['equal']
report={'status':'PASS_FIXED_HEAD_SENSITIVITY_AUDIT','scope':'same selected432validation records; frozen M10 heads, controls intentionally alter input consistency; not independent method performance',
 'conditions':result,'shuffle_summary':{'n':20,'mean_RMSE_Wm2':float(sh.mean()),'sample_std':float(sh.std(ddof=1)),'min_RMSE':float(sh.min()),'max_RMSE':float(sh.max())},
 'canary_independent_feature_comparison':check,'test_payloads_read':0,
 'interpretation':'Aligned predicted COT affects this frozen head; large mean/shuffle penalties are not deployable gains and do not isolate physical semantics,16x16context,supervision or effective capacity.'}
(B/'COUNTERFACTUAL_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ('status','shuffle_summary','canary_independent_feature_comparison')},indent=2))
