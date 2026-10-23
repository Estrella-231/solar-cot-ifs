import json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;result=json.loads((P/'RESULTS.json').read_text());out={}
for horizon in ['all','1h','2h','3h','4h']:
    days=result['summary'][horizon]['dates'];stats=np.zeros((len(days),3,3,5,2))
    for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
        for sj,seed in enumerate([42,43,44]):
            with np.load(P/'replay'/st/f'seed{seed}_Cselect.npz') as z:d={k:z[k] for k in z.files}
            mask=np.ones(len(d['truth']),bool)
            if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
            for di,day in enumerate(days):
                m=mask&(d['date']==day)
                for ai in range(5):stats[di,si,sj,ai]=[math.fsum(float(x)**2 for x in d['prediction'][ai,m]-d['truth'][m]),int(m.sum())]
    total=stats.sum(0);mse=total[...,0]/total[...,1];interaction=(mse[:,:,4]-mse[:,:,3])-(mse[:,:,2]-mse[:,:,1])
    weights=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);boot=[]
    for w in weights:
        t=np.tensordot(w,stats,axes=(0,0));v=t[...,0]/t[...,1];boot.append(((v[:,:,4]-v[:,:,3])-(v[:,:,2]-v[:,:,1])).mean())
    out[horizon]=dict(point=float(interaction.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),units='W2 m-4',by_station=dict(zip(['Niushou','Wangzhai','Mizhuang'],interaction.mean(1).tolist())))
(P/'MSE_INTERACTION.json').write_text(json.dumps(out,indent=2));print(out['all'])
