"""Read-only cold prediction witnesses on causal inputs from the original pack."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PRE=P.parent/'hubei_forecast_domain_cot_20261023';root=Path(json.loads((PRE/'PACK_COMPLETE.json').read_text())['path']);norm=json.loads((root/'norm.json').read_text());mu=np.array(norm['mean']);sd=np.array(norm['std']);X=np.load(root/'x.npy',mmap_mode='r');available=np.load(root/'forecast_mask.npy',mmap_mode='r')
rows=list(csv.DictReader((P/'OOF_predictions.csv').open()));ids=np.random.default_rng(123).choice(len(rows),130,replace=False);models={};maxdiff=0.;centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
for i in ids:
    r=rows[i];fold=int(r['fold']);field=int(r['field_index']);cy,cx=centers[r['station']];sl=(slice(cy-1,cy+2),slice(cx-1,cx+2));xx=np.array(X[field,:,sl[0],sl[1]],dtype='float64');xx[:25]=xx[:25]*sd[:,None,None]+mu[:,None,None];xx[17:25]=np.where(xx[25:33]>.5,xx[17:25],0);xx[16]=np.where(available[field,0,sl[0],sl[1]],xx[16],0);xx=xx.reshape(306)
    if fold not in models:
        with np.load(P/'models'/f'fold_{fold}.npz') as z:models[fold]={k:z[k] for k in ['mean','std','coef']}
    m=models[fold];out=np.concatenate([[1.],(xx-m['mean'])/m['std']])@m['coef'];expected=np.array([float(r['score_down']),float(r['score_stable']),float(r['score_up'])]);maxdiff=max(maxdiff,float(abs(out-expected).max()))
assert maxdiff<1e-10,maxdiff
report=dict(state='PASS_CAUSAL_INPUT_COLD_LINEAR_PROBE_REPLAY',witnesses=len(ids),max_abs_score_difference=maxdiff,GHI_values_read=0,real_future_input_read=0)
(P/'COLD_REPLAY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
