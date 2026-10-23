"""Score saved OOF directions; no fitting or threshold selection."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;z=np.load(P/'OOF_scores.npz');y=z['truth'];pred=z['prediction'];date=z['date'];station=z['station'];fold=z['fold']
def matrix(m,p=pred):return np.bincount(y[m]*3+p[m],minlength=9).reshape(3,3)
def met(cm):
    actual=cm.sum(1);forecast=cm.sum(0);recall=np.divide(np.diag(cm),actual,out=np.full(3,np.nan),where=actual>0);precision=np.divide(np.diag(cm),forecast,out=np.full(3,np.nan),where=forecast>0)
    return dict(n=int(cm.sum()),confusion=cm.tolist(),class_counts=actual.tolist(),predicted_counts=forecast.tolist(),recall=recall.tolist(),precision=[None if not np.isfinite(x) else float(x) for x in precision],balanced_accuracy=float(np.nanmean(recall)),accuracy=float(np.trace(cm)/cm.sum()),class_coverage=int((actual>0).sum()))
allmask=np.ones(len(y),dtype=bool);overall=met(matrix(allmask));by_site={s:met(matrix(station==s)) for s in ['Niushou','Wangzhai','Mizhuang']};by_fold={str(f):met(matrix(fold==f)) for f in range(3)}
dates=sorted(set(date));sums=np.stack([matrix(date==d) for d in dates]);w=np.random.default_rng(20261023).multinomial(len(dates),np.ones(len(dates))/len(dates),5000);boot=np.tensordot(w,sums,axes=(1,0));bacc=(np.diagonal(boot,axis1=1,axis2=2)/boot.sum(2)).mean(1)
precision_up=boot[:,2,2]/boot[:,:,2].sum(1);recall_up=boot[:,2,2]/boot[:,2,:].sum(1)
rows=[dict(date=d,**met(cm)) for d,cm in zip(dates,sums)]
with (P/'daily_metrics.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
constant={}
for label in range(3):constant[str(label)]=met(matrix(allmask,np.full(len(y),label)))
result=dict(state='COT_DIRECTION_OOF_DIAGNOSTIC_SCORED',overall=overall,station=by_site,fold=by_fold,constant_class_baselines=constant,balanced_accuracy_ci95=np.quantile(bacc,[.025,.975]).tolist(),up_precision_ci95=np.quantile(precision_up,[.025,.975]).tolist(),up_recall_ci95=np.quantile(recall_up,[.025,.975]).tolist(),dates=len(dates),bootstrap_draws=5000,scope='F_fit date-block OOF probe only, frozen upstream sources; not independent end-to-end test, COT RMSE gain or GHI gain',F_select_C_select_outer_payloads_read=0)
(P/'SCORES.json').write_text(json.dumps(result,indent=2));np.savez_compressed(P/'DATE_CONFUSIONS.npz',confusion=sums,dates=np.array(dates))
print(json.dumps({k:v for k,v in result.items() if k not in ['constant_class_baselines','station','fold']},indent=2))
