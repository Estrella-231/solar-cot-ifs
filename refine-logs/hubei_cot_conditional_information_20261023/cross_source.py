import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;O=P/'probe_attempt1';out=P/'cross_source_attempt1';out.mkdir(exist_ok=False)
arms=['G1','context','forecast_train_forecast_infer','observed_train_observed_infer_OFFLINE','observed_train_forecast_infer','forecast_train_observed_infer_OFFLINE'];daily={};records=[];replay_max=0.
for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
    with np.load(P/'inputs_attempt2'/f'{st}.npz') as z:d={k:z[k] for k in z.files}
    dest=out/st;dest.mkdir()
    for sj,seed in enumerate([42,43,44]):
        with np.load(O/st/f'seed{seed}_OOF.npz') as z:prior={k:z[k] for k in z.files}
        base=d['bases'][sj];x=np.column_stack([d['features'],base/d['clear']]);actual=x.copy();actual[:,279:297]=d['observed_COT_features'];pred=np.full((6,len(base)),np.nan);pred[:4]=prior['prediction']
        for fold in range(3):
            m=prior['probe_fold']==fold
            for ai,name,source in [(4,'observed_R_OFFLINE',x),(5,'forecast_COT',actual)]:
                with np.load(O/st/f'seed{seed}_{name}_fold{fold}.npz') as z:fit={k:z[k] for k in z.files}
                y=np.maximum(0,base[m]+1000*(float(fit['intercept'])+(source[m]-fit['mean'])/fit['scale']@fit['coefficient']));pred[ai,m]=np.where(d['available'][m],y,base[m])
                same=actual if name=='observed_R_OFFLINE' else x
                q=np.maximum(0,base[m]+1000*(float(fit['intercept'])+(same[m]-fit['mean'])/fit['scale']@fit['coefficient']));q=np.where(d['available'][m],q,base[m]);index=3 if name=='observed_R_OFFLINE' else 2
                replay_max=max(replay_max,float(abs(q-prior['prediction'][index,m]).max()))
        assert np.isfinite(pred).all()
        np.savez_compressed(dest/f'seed{seed}_OOF.npz',prediction=pred,truth=d['truth'],clear=d['clear'],arms=arms,date=prior['date'],probe_fold=prior['probe_fold'],init_time=d['init_time'],target_time=d['target_time'],lead=d['lead'],available=d['available'])
        for horizon in ['all','1h','2h','3h','4h']:
            mask=np.ones(len(base),bool)
            if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
            for ai,arm in enumerate(arms):
                e=pred[ai]-d['truth'];v=e[mask];records.append(dict(station=st,base_seed=seed,horizon=horizon,arm=arm,n=len(v),rmse=float(np.sqrt((v*v).mean())),mae=float(abs(v).mean()),bias=float(v.mean())))
                for day in sorted(set(prior['date'])):
                    m=mask&(prior['date']==day);daily.setdefault((horizon,day),np.zeros((3,3,6,2)))[si,sj,ai]=[float(e[m]@e[m]),int(m.sum())]
assert replay_max<1e-10;summary={};saves={}
for horizon in ['all','1h','2h','3h','4h']:
    days=sorted(k[1] for k in daily if k[0]==horizon);stats=np.stack([daily[horizon,day] for day in days]);saves[horizon]=stats;total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]);mean=point.mean((0,1));w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);den=w@stats[...,1].reshape(len(days),-1);assert (den>0).all();boot=np.sqrt((w@stats[...,0].reshape(len(days),-1))/den).reshape(-1,3,3,6).mean((1,2));comparisons={}
    for j,k in [(4,1),(4,2),(4,0),(5,2),(3,2)]:comparisons[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(mean[j]-mean[k]),ci95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist(),relative_change_percent=float((mean[j]/mean[k]-1)*100))
    summary[horizon]=dict(rmse=dict(zip(arms,mean.tolist())),comparisons=comparisons,by_station={st:dict(zip(arms,point[si].mean(0).tolist())) for si,st in enumerate(['Niushou','Wangzhai','Mizhuang'])},dates=days,rows=int(total[:,0,0,1].sum()))
with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
np.savez_compressed(out/'DATE_SUMS.npz',**saves,arms=arms)
(out/'RESULTS.json').write_text(json.dumps(dict(state='COLD_CROSS_SOURCE_COMPLETE',summary=summary,replay_max=replay_max,new_fits=0,test_payloads_read=0,claim='Posthoc train-date exploration; only forecast-infer arms use causal inputs'),indent=2));print(json.dumps(summary['all'],indent=2))
