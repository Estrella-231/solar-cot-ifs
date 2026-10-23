import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;O=P/'probe_attempt1';assert json.loads((O/'COMPLETE.json').read_text())['state']=='FIXED_DATE_OOF_RIDGE_COMPLETE'
sites=['Niushou','Wangzhai','Mizhuang'];arms=['G1','context','forecast_COT','observed_R_OFFLINE'];rows=[];daily={};error_max=0.
for si,st in enumerate(sites):
    for sj,seed in enumerate([42,43,44]):
        with np.load(O/st/f'seed{seed}_OOF.npz') as z:d={k:z[k] for k in z.files}
        assert d['arms'].tolist()==arms
        for horizon in ['all','1h','2h','3h','4h']:
            mask=np.ones(len(d['truth']),bool)
            if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
            for ai,arm in enumerate(arms):
                e=d['prediction'][ai]-d['truth'];v=e[mask];rmse=math.sqrt(math.fsum(float(x)**2 for x in v)/len(v));error_max=max(error_max,abs(rmse-float(np.sqrt((v*v).mean()))))
                rows.append(dict(station=st,base_seed=seed,horizon=horizon,arm=arm,n=len(v),rmse=rmse,mae=float(abs(v).mean()),bias=float(v.mean())))
                for day in sorted(set(d['date'])):
                    m=mask&(d['date']==day);daily.setdefault((horizon,day),np.zeros((3,3,4,2)))[si,sj,ai]=[float(e[m]@e[m]),int(m.sum())]
summary={};saves={}
for horizon in ['all','1h','2h','3h','4h']:
    days=sorted(k[1] for k in daily if k[0]==horizon);stats=np.stack([daily[horizon,day] for day in days]);saves[horizon]=stats;total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]);mean=point.mean((0,1))
    w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);den=w@stats[...,1].reshape(len(days),-1);assert (den>0).all();boot=np.sqrt((w@stats[...,0].reshape(len(days),-1))/den).reshape(-1,3,3,4).mean((1,2));comparison={}
    for j,k in [(2,1),(3,1),(1,0),(2,0),(3,0)]:comparison[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(mean[j]-mean[k]),relative_change_percent=float((mean[j]/mean[k]-1)*100),ci95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist())
    summary[horizon]=dict(rmse=dict(zip(arms,mean.tolist())),comparisons=comparison,by_station={st:dict(zip(arms,point[si].mean(0).tolist())) for si,st in enumerate(sites)},basewise_forecast_minus_context=(point[:,:,2]-point[:,:,1]).mean(0).tolist(),dates=days,rows=int(total[:,0,0,1].sum()))
main=summary['all'];primary=main['comparisons']['forecast_COT_minus_context'];conditions=dict(point_improves=primary['delta_rmse']<0,ci_below_zero=primary['ci95'][1]<0,station_guard=bool(np.all([v['forecast_COT']<=1.01*v['context'] for v in main['by_station'].values()])),base_direction_guard=sum(x<0 for x in main['basewise_forecast_minus_context'])>=2)
with (P/'metrics.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(P/'DATE_SUMS.npz',**saves,arms=arms)
(P/'RESULTS.json').write_text(json.dumps(dict(state='SCORED',summary=summary,conditions=conditions,continue_probe=bool(np.all(list(conditions.values()))),fsum_max_difference=error_max,C_select_scored_rows=0,test_payloads_read=0),indent=2));print(json.dumps(dict(all=main,conditions=conditions),indent=2))
