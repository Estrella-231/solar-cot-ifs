"""Reproduce validation strata and post-hoc early-checkpoint diagnostic."""
from pathlib import Path
import json
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent/'results_attempt2';pack=np.load(R/'paired_pack.npz')
pred={a:np.stack([np.load(R/f'{a}_seed{s}/predictions.npz')['pred_ghi'] for s in (42,43,44)]) for a in ('A','transport','hybrid')}
ref=np.load(R/'A_seed42/predictions.npz');y=ref['true_ghi'];lead=pack['val_lead'];idx=np.arange(len(y))
cov=pack['val_support'][idx,8+lead].mean((1,2));cot=np.expm1(pack['val_cot_transport'][idx,8+lead,1,1]);kt=y/pack['val_clear']
def metric(mask):
 return {a:dict(rmse=float(np.sqrt(((v[:,mask]-y[mask])**2).mean(1)).mean()),bias=float((v[:,mask]-y[mask]).mean()),mae=float(abs(v[:,mask]-y[mask]).mean())) for a,v in pred.items()}
groups={'all':np.ones(len(y),bool),'support_full':cov==1,'support_zero':cov==0,'support_partial':(cov>0)&(cov<1),'kt_ge_09':kt>=.9,'kt_lt_04':kt<.4,'cot_ge30':cot>=30,'cot_lt01':cot<.1}
strata={k:dict(n=int(m.sum()),metrics=metric(m)) for k,m in groups.items() if m.any()}
(D/'global_decomposition.json').write_text(json.dumps(strata,indent=2),encoding='utf-8')
contributions={}
for arm in ('transport','hybrid'):
 delta=((pred[arm]-y)**2-(pred['A']-y)**2).mean(0)
 contributions[arm]=dict(net_delta_squared_error=float(delta.sum()),high_kt_fraction_of_net_delta=float(delta[kt>=.9].sum()/delta.sum()),low_kt_fraction_of_net_delta=float(delta[kt<.4].sum()/delta.sum()),fraction_rows_with_lower_seed_mean_squared_error=float((delta<0).mean()))
(D/'error_contributions.json').write_text(json.dumps(contributions,indent=2),encoding='utf-8')
days=ref['target_day'];unique=np.unique(days);rng=np.random.default_rng(20261015);draw=rng.integers(len(unique),size=(2000,len(unique)));w=np.stack([np.bincount(r,minlength=len(unique)) for r in draw]);N=np.array([(days==d).sum() for d in unique]);boots={};points={};chosen=[];preds={}
for arm in ('A','transport','hybrid'):
 es=[];pr=[]
 for s in (42,43,44):
  with np.load(R/f'{arm}_seed{s}/predictions.npz') as old:best=old['pred_ghi'].copy()
  meta=json.loads((R/f'{arm}_seed{s}/complete.json').read_text());step=meta['selected_step'];origin='original_best';score=np.sqrt(((best-y)**2).mean())
  for f in (D/'early_replay_attempt1'/f'{arm}_seed{s}').glob('pred_step*.npz'):
   with np.load(f) as z:
    assert np.array_equal(ref['row_ids'],z['row_ids']);v=z['pred_ghi'];cur=np.sqrt(((v-y)**2).mean())
    if cur<score:best=v.copy();score=cur;step=int(f.stem[-3:]);origin='early_replay'
  chosen.append(dict(arm=arm,seed=s,step=step,origin=origin,rmse=float(score)));pr.append(best);es.append(best-y)
 e=np.stack(es);preds[arm]=np.stack(pr);points[arm]=float(np.sqrt((e*e).mean(1)).mean());sq=np.stack([(e[:,days==d]**2).sum(1) for d in unique]);boots[arm]=np.sqrt((w@sq)/(w@N)[:,None]).mean(1)
results={a:dict(A_rmse=points['A'],AC_rmse=points[a],delta=points[a]-points['A'],reduction_percent=100*(points['A']-points[a])/points['A'],ci95=np.quantile(boots[a]-boots['A'],[.025,.975]).tolist()) for a in ('transport','hybrid')}
(D/'early_selection_diagnostic.json').write_text(json.dumps(dict(scope='post-hoc early-checkpoint diagnostic; not independent performance confirmation',step100_replay_all_exact=True,selection=chosen,results=results),indent=2),encoding='utf-8')
np.savez_compressed(D/'early_selected_predictions.npz',**preds,true_ghi=y,row_ids=ref['row_ids'],lead_minutes=ref['lead_minutes'],target_day=days)
print(json.dumps(results,indent=2))
