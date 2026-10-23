"""Physical GHI paired development scoring; no checkpoint selection."""
import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;G=P.parent/'hubei_cot_calibration_coverage_20261023'
assert json.loads((P/'replay/COMPLETE.json').read_text())['state']=='PAIRED_DUAL_DOMAIN_ALL_COMPLETE'
arms=['G1','forecast_no_COT','forecast_future_COT','dual_no_COT','dual_future_COT'];sites=['Niushou','Wangzhai','Mizhuang'];records=[];daily={};meta={};checks=[]
for si,st in enumerate(sites):
    folder=P/'replay'/st;receipt=json.loads((folder/'COMPLETE.json').read_text());old_receipt=json.loads((G/'replay'/st/'COMPLETE.json').read_text());assert receipt['state']=='PAIRED_DUAL_DOMAIN_FOUR_ARM_COMPLETE'
    for seed,new,old in zip([42,43,44],receipt['receipts'],old_receipt['receipts']):
        original={a['arm']:a for a in old['arms']}
        for a in new['arms']:
            control=original['large_no_future_COT' if a['arm'].endswith('no_COT') else 'large_future_COT']
            assert a['initialization_sha256']==control['initialization_sha256'] and a['order_sha256']==control['order_sha256']
            assert a['updates']==60 and a['sampled_examples']==92160 and a['parameters']==9922
        with np.load(folder/f'seed{seed}_Cselect.npz') as z:d={k:z[k] for k in z.files}
        assert d['arms'].tolist()==arms
        with np.load(G/'replay'/st/f'seed{seed}_Cselect.npz') as z:old={k:z[k] for k in z.files}
        for k in ['init_time','target_time','lead','truth','clear']:assert np.array_equal(d[k],old[k]),k
        assert len(set(zip(d['init_time'],d['target_time'],d['lead'])))==len(d['truth'])
        maxerr=max(float(abs(d['prediction'][i]-old['prediction'][j]).max()) for i,j in [(0,0),(1,3),(2,4)]);assert maxerr<.001
        checks.append(dict(station=st,seed=seed,original_prediction_max_error=maxerr))
        for horizon in ['all','1h','2h','3h','4h','high_GHI_ge800']:
            mask=np.ones(len(d['truth']),bool)
            if horizon.endswith('h') and horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
            if horizon=='high_GHI_ge800':mask&=d['truth']>=800
            for ai,arm in enumerate(arms):
                e=d['prediction'][ai]-d['truth'];v=e[mask];n=len(v)
                rmse=math.sqrt(math.fsum(float(x)**2 for x in v)/n) if n else None
                vector=float(np.sqrt((v*v).mean())) if n else None
                assert n==0 or abs(rmse-vector)<1e-10
                records.append(dict(station=st,seed=seed,horizon=horizon,arm=arm,n=n,rmse=rmse,mae=float(abs(v).mean()) if n else None,bias=float(v.mean()) if n else None))
                if horizon!='high_GHI_ge800':
                    for day in sorted(set(d['date'])):
                        m=mask&(d['date']==day);daily.setdefault((horizon,day),np.zeros((3,3,5,2)))[si,seed-42,ai]=[float(e[m]@e[m]),int(m.sum())]
summaries={};sums={}
for horizon in ['all','1h','2h','3h','4h']:
    days=sorted({k[1] for k in daily if k[0]==horizon});stats=np.stack([daily[horizon,day] for day in days]);sums[horizon]=stats
    total=stats.sum(0);points=np.sqrt(total[...,0]/total[...,1]);w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
    den=w@stats[...,1].reshape(len(days),-1);assert (den>0).all();b=np.sqrt((w@stats[...,0].reshape(len(days),-1))/den).reshape(-1,3,3,5).mean((1,2));mean=points.mean((0,1));comparison={}
    for j,k in [(4,3),(2,1),(4,2),(3,1),(4,0)]:
        comparison[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(mean[j]-mean[k]),ci95=np.quantile(b[:,j]-b[:,k],[.025,.975]).tolist(),relative_change_percent=float((mean[j]/mean[k]-1)*100))
    summaries[horizon]=dict(rmse=dict(zip(arms,mean.tolist())),comparison=comparison,by_station_rmse={st:dict(zip(arms,points[si].mean(0).tolist())) for si,st in enumerate(sites)},seed_equal_station_delta=(points[:,:,4]-points[:,:,3]).mean(0).tolist(),dates=days,rows=int(total[:,0,0,1].sum()))
high=[]
for st in sites:
    r=[x for x in records if x['station']==st and x['horizon']=='high_GHI_ge800']
    no=[x['bias'] for x in r if x['arm']==arms[3] and x['bias'] is not None];reg=[x['bias'] for x in r if x['arm']==arms[4] and x['bias'] is not None]
    high.append(dict(station=st,rows=next(x['n'] for x in r),absolute_bias_increase=float(abs(np.mean(reg))-abs(np.mean(no))) if no else None))
main=summaries['all'];primary=main['comparison'][f'{arms[4]}_minus_{arms[3]}']
# Explicit named conditions; no selection after examining these results.
conditions=dict(point_improves=primary['delta_rmse']<0,ci_below_zero=primary['ci95'][1]<0,
    station_guard=bool(np.all([v[arms[4]]<=1.01*v[arms[3]] for v in main['by_station_rmse'].values()])),
    seeds_guard=sum(x<0 for x in main['seed_equal_station_delta'])>=2,
    high_GHI_guard=bool(np.all([x['absolute_bias_increase'] is not None and x['absolute_bias_increase']<=5 for x in high])))
with (P/'metrics.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
np.savez_compressed(P/'DATE_ERROR_SUMS.npz',**sums,arms=arms)
out=dict(state='SCORED',summary=summaries,high_GHI_guard=high,conditions=conditions,continue_candidate=bool(np.all(list(conditions.values()))),checks=checks,independent_point_scoring='math.fsum vs numpy per station/seed/arm',test_payloads_read=0,claim='Repeated training-internal development dates; not independent forecast confirmation')
(P/'RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(all=main,conditions=conditions,high_GHI=high),indent=2))
