from pathlib import Path
import numpy as np,pandas as pd,json
P=Path(__file__).parent;ROOT=P.parents[2]
B=ROOT/'refine-logs/hubei_same_time_geometry_20261102/recovered'
rows=pd.read_csv(B/'pack/rows.csv',dtype={'time_bjt':str})
im=np.load(P/'validation_target_images.npz');lookup={t:i for i,t in enumerate(im['times_bjt'])}
ST={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
frames=[]
for st,(y,x) in ST.items():
    pred=[]
    first_ids=None;first_truth=None
    for seed in [42,43,44]:
        d=B/'training'/f'{st}_seed{seed}'
        a=np.load(d/'A_validation.npz');c=np.load(d/'AC_validation.npz')
        assert np.array_equal(a['global_indices'],c['global_indices'])
        assert np.array_equal(a['truth'],c['truth'])
        if first_ids is None:first_ids=a['global_indices'].copy();first_truth=a['truth'].copy()
        else:
            assert np.array_equal(a['global_indices'],first_ids)
            assert np.array_equal(a['truth'],first_truth)
        pred.append(np.stack([a['prediction'],c['prediction']]))
    q=rows.iloc[a['global_indices']].copy();q['truth']=a['truth'];p=np.stack(pred).astype(float)
    q['A']=p[:,0].mean(0);q['AC']=p[:,1].mean(0)
    q['A_mse_seedmean']=((p[:,0]-a['truth'])**2).mean(0)
    q['AC_mse_seedmean']=((p[:,1]-a['truth'])**2).mean(0)
    for j,seed in enumerate([42,43,44]):q[f'A_seed{seed}']=p[j,0];q[f'AC_seed{seed}']=p[j,1]
    q['date']=q['time_bjt'].str[:8];q['time']=pd.to_datetime(q['time_bjt'],format='%Y%m%d%H%M')
    q['COT']=[float(np.expm1(im['cot_log1p'][lookup[t],y,x])) for t in q['time_bjt']]
    frames.append(q)
allrows=pd.concat(frames);allrows.to_csv(P/'current_validation_predictions.csv',index=False)
out=[]
for (st,date),g in allrows.groupby(['station','date'],sort=True):
    g=g.sort_values('time');w=g[(g.time.dt.hour*60+g.time.dt.minute>=630)&(g.time.dt.hour*60+g.time.dt.minute<=870)]
    a=float(np.mean([np.sqrt(np.mean((w[f'A_seed{s}']-w.truth)**2)) for s in [42,43,44]]))
    c=float(np.mean([np.sqrt(np.mean((w[f'AC_seed{s}']-w.truth)**2)) for s in [42,43,44]]))
    rec={'station':st,'date':date,'n':len(w),'A_rmse':a,'AC_rmse':c,'gain_pct':100*(a-c)/a,'cot_min':w.COT.min(),'cot_max':w.COT.max(),'cot_range':w.COT.max()-w.COT.min(),'ghi_min':w.truth.min(),'ghi_max':w.truth.max()}
    # Current sampling has missing scored targets on every date. Retain gaps;
    # require at least 12 of the 17 fixed midday targets, without interpolating.
    # The 30 threshold is retrieved COT, not a CPP-reference thickness label.
    rec['eligible']=bool(len(w)>=12 and rec['cot_max']>=30 and rec['cot_range']>=5)
    out.append(rec)
pd.DataFrame(out).to_csv(P/'event_candidates.csv',index=False)
print(pd.DataFrame(out).query('eligible').sort_values(['date','station']).to_string(index=False))
print('all physical RGB quantiles',np.quantile(im['raw'][:,:3],[0,.01,.5,.99,1]))
print('physical COT range',np.expm1(im['cot_log1p']).min(),np.expm1(im['cot_log1p']).max())
