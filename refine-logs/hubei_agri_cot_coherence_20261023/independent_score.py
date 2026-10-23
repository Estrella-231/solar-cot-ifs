import json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;result=json.loads((P/'RESULTS.json').read_text());pointmax=0.;cimax=0.
for scope in ['all','common']:
    for horizon in ['all','1h','2h','3h','4h']:
        expected=result['summary'][scope][horizon];days=expected['dates'];stats=np.zeros((len(days),3,3,8,2))
        for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
            for sj,seed in enumerate([42,43,44]):
                with np.load(P/'replay'/st/f'seed{seed}_factorial.npz') as z:d={k:z[k] for k in z.files}
                mask=np.ones(len(d['truth']),bool)
                if scope=='common':mask&=d['common']>0
                if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
                for di,day in enumerate(days):
                    m=mask&(d['date']==day)
                    for ai in range(8):stats[di,si,sj,ai]=[math.fsum(float(x)**2 for x in d['prediction'][ai,m]-d['truth'][m]),int(m.sum())]
        total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]).mean((0,1));arms=list(expected['rmse'])
        pointmax=max(pointmax,max(abs(point[i]-expected['rmse'][arm]) for i,arm in enumerate(arms)))
        weights=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);boot=[];inter=[]
        for w in weights:
            t=np.tensordot(w,stats,axes=(0,0));mse=t[...,0]/t[...,1];boot.append(np.sqrt(mse).mean((0,1)));inter.append(((mse[:,:,7]-mse[:,:,6])-(mse[:,:,3]-mse[:,:,2])).mean())
        boot=np.array(boot)
        for j,k in [(3,2),(7,6),(2,1),(7,5),(6,5),(5,1),(4,0)]:
            ci=np.quantile(boot[:,j]-boot[:,k],[.025,.975]);cimax=max(cimax,float(abs(ci-expected['comparisons'][f'{arms[j]}_minus_{arms[k]}']['ci95']).max()))
        ci=np.quantile(inter,[.025,.975]);cimax=max(cimax,float(abs(ci-expected['COT_source_MSE_interaction']['ci95']).max()))
assert pointmax<1e-10 and cimax<1e-9
(P/'INDEPENDENT_SCORE.json').write_text(json.dumps(dict(state='PASS',max_point_difference=pointmax,max_ci_difference=cimax,new_fits=0,test_payloads_read=0),indent=2));print(pointmax,cimax)
