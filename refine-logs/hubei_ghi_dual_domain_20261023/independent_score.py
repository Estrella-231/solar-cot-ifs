"""Independent date accumulators and paired bootstrap, saved predictions only."""
import json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;report=json.loads((P/'RESULTS.json').read_text());maximum=0.;ci_max=0.
for horizon in ['all','1h','2h','3h','4h']:
    days=report['summary'][horizon]['dates'];stats=np.zeros((len(days),3,3,5,2))
    for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
        for sj,seed in enumerate([42,43,44]):
            with np.load(P/'replay'/st/f'seed{seed}_Cselect.npz') as z:d={k:z[k] for k in z.files}
            mask=np.ones(len(d['truth']),bool)
            if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
            for di,day in enumerate(days):
                m=mask&(d['date']==day)
                for ai in range(5):stats[di,si,sj,ai]=[math.fsum(float(x)**2 for x in (d['prediction'][ai,m]-d['truth'][m])),int(m.sum())]
    total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]).mean((0,1))
    expected=report['summary'][horizon];arms=list(expected['rmse']);maximum=max(maximum,max(abs(point[i]-expected['rmse'][a]) for i,a in enumerate(arms)))
    rng=np.random.default_rng(20261023);weights=rng.multinomial(len(days),np.ones(len(days))/len(days),5000);samples=[]
    for w in weights:
        v=np.tensordot(w,stats,axes=(0,0));samples.append(np.sqrt(v[...,0]/v[...,1]).mean((0,1)))
    samples=np.array(samples)
    for j,k in [(4,3),(2,1),(4,2),(3,1),(4,0)]:
        ci=np.quantile(samples[:,j]-samples[:,k],[.025,.975]);old=expected['comparison'][f'{arms[j]}_minus_{arms[k]}']['ci95'];ci_max=max(ci_max,float(abs(ci-old).max()))
assert maximum<1e-10 and ci_max<1e-10
(P/'INDEPENDENT_SCORE.json').write_text(json.dumps(dict(state='PASS',max_point_difference=maximum,max_ci_difference=ci_max,test_payloads_read=0),indent=2))
print(maximum,ci_max)
