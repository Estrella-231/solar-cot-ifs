import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;assert json.loads((P/'replay/COMPLETE.json').read_text())['state']=='FROZEN_AGRI_COT_FACTORIAL_ALL_COMPLETE'
sites=['Niushou','Wangzhai','Mizhuang'];rows=[];acc={};coverage={};error_max=0.
for si,st in enumerate(sites):
    receipt=json.loads((P/'replay'/st/'COMPLETE.json').read_text());assert receipt['state']=='FROZEN_AGRI_COT_FACTORIAL_COMPLETE';coverage[st]={k:receipt[k] for k in ['rows','common','geometry_max']}
    prior=None
    for sj,seed in enumerate([42,43,44]):
        with np.load(P/'replay'/st/f'seed{seed}_factorial.npz') as z:d={k:z[k] for k in z.files}
        arms=d['arms'].tolist();keys=list(zip(d['init_time'],d['target_time'],d['lead']));assert len(set(keys))==len(keys)
        if prior is not None:assert keys==prior[0] and np.array_equal(d['truth'],prior[1])
        prior=keys,d['truth']
        for scope,scope_mask in [('all',np.ones(len(keys),bool)),('common',d['common']>0)]:
            for horizon in ['all','1h','2h','3h','4h']:
                mask=scope_mask.copy()
                if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
                for ai,arm in enumerate(arms):
                    e=d['prediction'][ai]-d['truth'];v=e[mask];rmse=math.sqrt(math.fsum(float(x)**2 for x in v)/len(v));error_max=max(error_max,abs(rmse-float(np.sqrt((v*v).mean()))))
                    rows.append(dict(scope=scope,horizon=horizon,station=st,seed=seed,arm=arm,n=len(v),rmse=rmse,mae=float(abs(v).mean()),bias=float(v.mean())))
                    for day in sorted(set(d['date'])):
                        m=mask&(d['date']==day);acc.setdefault((scope,horizon,day),np.zeros((3,3,8,2)))[si,sj,ai]=[float(e[m]@e[m]),int(m.sum())]
assert error_max<1e-10
summary={};saves={}
for scope in ['all','common']:
    summary[scope]={}
    for horizon in ['all','1h','2h','3h','4h']:
        days=sorted(k[2] for k in acc if k[:2]==(scope,horizon));stats=np.stack([acc[scope,horizon,d] for d in days]);saves[scope+'_'+horizon]=stats
        total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]);w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
        den=w@stats[...,1].reshape(len(days),-1);assert (den>0).all();mse=((w@stats[...,0].reshape(len(days),-1))/den).reshape(-1,3,3,8);boot=np.sqrt(mse).mean((1,2));mean=point.mean((0,1));comparisons={}
        for j,k in [(3,2),(7,6),(2,1),(7,5),(6,5),(5,1),(4,0)]:comparisons[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(mean[j]-mean[k]),ci95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist())
        interaction=(total[:,:,7,0]/total[:,:,7,1]-total[:,:,6,0]/total[:,:,6,1])-(total[:,:,3,0]/total[:,:,3,1]-total[:,:,2,0]/total[:,:,2,1])
        bootstrap_interaction=((mse[:,:,:,7]-mse[:,:,:,6])-(mse[:,:,:,3]-mse[:,:,:,2])).mean((1,2))
        summary[scope][horizon]=dict(rmse=dict(zip(arms,mean.tolist())),comparisons=comparisons,
            by_station={st:dict(zip(arms,point[si].mean(0).tolist())) for si,st in enumerate(sites)},
            COT_source_MSE_interaction=dict(point=float(interaction.mean()),ci95=np.quantile(bootstrap_interaction,[.025,.975]).tolist(),by_station=dict(zip(sites,interaction.mean(1).tolist()))),dates=days,rows=int(total[:,0,0,1].sum()))
with (P/'metrics.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(P/'DATE_SUMS.npz',**saves,arms=arms)
(P/'RESULTS.json').write_text(json.dumps(dict(state='SCORED',summary=summary,coverage=coverage,independent_point_max_difference=error_max,new_fits=0,test_payloads_read=0,claim='Observed future AGRI/COT branches are offline conditional diagnostics only'),indent=2))
print(json.dumps(summary['all']['all'],indent=2))
