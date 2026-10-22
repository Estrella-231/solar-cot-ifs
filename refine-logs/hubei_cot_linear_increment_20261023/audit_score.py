import json,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/'screen_attempt1';records=[];daily={};points=[]
expected=json.loads((O/'SCREEN_RESULT.json').read_text())
for i,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
 for j,seed in enumerate([42,43,44]):
  z=np.load(O/f'{st}_seed{seed}_Cselect.npz');pred=z['prediction'].astype(np.float64);truth=z['truth'].astype(np.float64);e=pred-truth
  assert pred.shape[0]==3 and z['coefficients'].shape==(2,27) and np.isfinite(pred).all()
  points.append(np.linalg.norm(e,axis=1)/np.sqrt(len(truth)))
  for day in set(z['date']):
   if day not in daily:daily[day]=np.zeros((3,3,3,2))
   m=z['date']==day;daily[day][i,j,:,0]=(e[:,m]*e[:,m]).sum(1);daily[day][i,j,:,1]=m.sum()
point=np.array(points).mean(0);wanted=np.array([expected['arms'][a] for a in ['A','H','F']]);error=float(abs(point-wanted).max());assert error<1e-9
arr=np.array([daily[d] for d in sorted(daily)]);w=np.random.default_rng(20261023).multinomial(len(daily),np.ones(len(daily))/len(daily),5000)
boot=np.sqrt(np.einsum('bd,dsja->bsja',w,arr[...,0])/np.einsum('bd,dsja->bsja',w,arr[...,1])).mean((1,2))
for a,b in [('F','A'),('F','H'),('H','A')]:
 i=['A','H','F'].index(a);j=['A','H','F'].index(b);ci=np.quantile(boot[:,i]-boot[:,j],[.025,.975]);diff=float(abs(ci-np.array(expected['comparisons'][a+'_minus_'+b]['ci95'])).max());assert diff<1e-9;error=max(error,diff)
report=dict(state='PASS_INDEPENDENT_CSELECT_SCORE',max_abs_difference=error,point=point.tolist(),dates=len(daily),test_payloads_read=0,official_validation_labels_read=0)
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2));print(report)
