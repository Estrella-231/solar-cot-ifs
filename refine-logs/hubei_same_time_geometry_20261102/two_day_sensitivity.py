from pathlib import Path
from datetime import datetime
import json,csv
import numpy as np
P=Path(__file__).parent;R=json.loads((P/'INDEPENDENT_FP64_RESULTS.json').read_text());dates=R['target_dates'];assert len(dates)%2==0
blocks=[]
for i in range(0,len(dates),2):
 assert (datetime.strptime(dates[i+1],'%Y%m%d')-datetime.strptime(dates[i],'%Y%m%d')).days==1
 blocks.append(dates[i:i+2])
rows=list(csv.DictReader((P/'recovered/pack/rows.csv').open()));counts=[];sums=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
 for seed in [42,43,44]:
  d=P/'recovered/training'/f'{st}_seed{seed}';a=np.load(d/'A_validation.npz');b=np.load(d/'AC_validation.npz');ds=np.array([rows[i]['time_bjt'][:8] for i in a['global_indices']]);err=np.stack([a['prediction'],b['prediction']]).astype(float)-a['truth'].astype(float)
  counts.append([np.isin(ds,bb).sum() for bb in blocks]);sums.append([[np.sum(err[arm,np.isin(ds,bb)]**2) for bb in blocks] for arm in [0,1]])
counts=np.array(counts);sums=np.array(sums);rng=np.random.default_rng(20261102);draw=rng.integers(0,len(blocks),size=(5000,len(blocks)));w=np.stack([(draw==i).sum(1) for i in range(len(blocks))],1);den=w@counts.T;boot=np.sqrt(np.stack([w@sums[:,i].T for i in [0,1]],2)/den[:,:,None]);mean=boot.mean(1)
out={'scope':'pre-existing consecutive two-day validation blocks; sensitivity to retaining adjacent-day dependence','blocks':blocks,'delta_ci95':np.quantile(mean[:,1]-mean[:,0],[.025,.975]).tolist(),'gain_ci95':np.quantile(100*(mean[:,0]-mean[:,1])/mean[:,0],[.025,.975]).tolist(),'point_gain_pct':R['gain_pct'],'test_payloads_read':0}
(P/'TWO_DAY_BLOCK_SENSITIVITY.json').write_text(json.dumps(out,indent=2));print(out['delta_ci95'],out['gain_ci95'])
