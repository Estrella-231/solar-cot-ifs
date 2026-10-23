import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;O=P/'training_attempt1';assert all(json.loads((O/st/'COMPLETE.json').read_text())['state']=='FIXED_600_UPDATE_OOF_COMPLETE' for st in ['Niushou','Wangzhai','Mizhuang'])
sites=['Niushou','Wangzhai','Mizhuang'];arms=['G1','physical','task','direct_AGRI'];rows=[];daily={};error_max=0.
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
    for j,k in [(2,1),(2,0),(2,3),(1,0),(3,0)]:comparison[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(mean[j]-mean[k]),relative_change_percent=float((mean[j]/mean[k]-1)*100),ci95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist())
    summary[horizon]=dict(rmse=dict(zip(arms,mean.tolist())),comparisons=comparison,by_station={st:dict(zip(arms,point[si].mean(0).tolist())) for si,st in enumerate(sites)},basewise_task_minus_physical=(point[:,:,2]-point[:,:,1]).mean(0).tolist(),dates=days,rows=int(total[:,0,0,1].sum()))
main=summary['all'];primary=main['comparisons']['task_minus_physical'];conditions=dict(point_improves=primary['delta_rmse']<0,ci_below_zero=primary['ci95'][1]<0,beats_G1_ci=main['comparisons']['task_minus_G1']['ci95'][1]<0,station_guard=bool(np.all([v['task']<=1.01*v['G1'] for v in main['by_station'].values()])),base_direction_guard=sum(x<0 for x in main['basewise_task_minus_physical'])>=2)
# Additional physical and high-GHI safeguards, same original availability.
protection=[]
for st in sites:
    for seed in [42,43,44]:
        with np.load(O/st/f'seed{seed}_OOF.npz') as z:d={k:z[k] for k in z.files}
        av=d['available'];ref=d['cot_reference'][av];co=d['cot_prediction'][:,av];raw=d['cot_raw'][av]
        rm=np.sqrt(((co-ref[None])**2).mean((1,2)))
        def far(v):
            predicted=v>=30;fp=(predicted & (ref<30)).sum();return float(fp/predicted.sum()) if predicted.sum() else 0.
        rawfar=far(raw);taskfar=far(co[1]);high=d['truth']>=800
        hb=np.abs((d['prediction'][:,high]-d['truth'][None,high]).mean(1)) if high.any() else None
        protection.append(dict(station=st,seed=seed,COT_rmse=rm.tolist(),raw_thick_FAR=rawfar,task_thick_FAR=taskfar,high_GHI_n=int(high.sum()),high_GHI_abs_bias=hb.tolist() if hb is not None else None,pass_COT=bool(rm[1]<=1.01*rm[0]),pass_FAR=taskfar<=rawfar+.02,pass_high=bool(hb[2]<=hb[0]+5) if hb is not None else True))
conditions.update(COT_guard=all(r['pass_COT'] for r in protection),FAR_guard=all(r['pass_FAR'] for r in protection),high_GHI_guard=all(r['pass_high'] for r in protection))
(P/'PROTECTION.json').write_text(json.dumps(protection,indent=2))
with (P/'metrics.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(P/'DATE_SUMS.npz',**saves,arms=arms)
(P/'RESULTS.json').write_text(json.dumps(dict(state='SCORED',summary=summary,conditions=conditions,continue_probe=bool(np.all(list(conditions.values()))),fsum_max_difference=error_max,C_select_scored_rows=0,test_payloads_read=0),indent=2));print(json.dumps(dict(all=main,conditions=conditions),indent=2))
