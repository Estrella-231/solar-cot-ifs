"""Describe source-change response; labels are used only for scoring groups."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
output=[]; rows=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    with np.load(P/'replay'/st/'physical_COT_sources.npz') as z:q={k:z[k] for k in z.files}
    ds=[]
    for seed in [42,43,44]:
        with np.load(P/'replay'/st/f'seed{seed}_response.npz') as z:d={k:z[k] for k in z.files}
        ds.append(d)
        m=d['joint_available']>0; before=d['prediction'][4];after=d['prediction'][6]
        shift=after-before;e=before-d['truth']; dc=q['observed_R'][:,4]-q['forecast'][:,4]
        changed=abs(shift)>1e-6
        groups=[('all',np.ones(len(m),bool)),('source_thickens_gt3',dc>3),('source_thins_lt_minus3',dc< -3),
                ('source_center_stable_3',abs(dc)<=3),('forecast_underestimates',e<0),('forecast_overestimates',e>0)]
        for name,g in groups:
            take=m&g;n=int(take.sum())
            if not n:continue
            active=take&changed
            rows.append(dict(station=st,seed=seed,group=name,n=n,changed_n=int(active.sum()),
                mean_source_COT_change=float(dc[take].mean()),mean_GHI_shift=float(shift[take].mean()),
                direction_toward_observation_fraction_on_changed=float((e[active]*shift[active]<0).mean()) if active.any() else None,
                opposite_center_COT_GHI_direction_fraction_on_changed=float((dc[active]*shift[active]<0).mean()) if active.any() else None,
                delta_sse=float(((after[take]-d['truth'][take])**2-e[take]**2).sum())))
    dates=sorted(set(ds[0]['date']));daily=[]
    for day in dates:
        m=(ds[0]['date']==day)&(ds[0]['joint_available']>0)
        diffs=[float(((d['prediction'][6,m]-d['truth'][m])**2-(d['prediction'][4,m]-d['truth'][m])**2).sum()) for d in ds]
        daily.append(dict(date=day,n=int(m.sum()),mean_seed_delta_sse=float(np.mean(diffs)),by_seed_delta_sse=diffs))
    loo=[]
    for omit in dates:
        m=(ds[0]['date']!=omit)&(ds[0]['joint_available']>0)
        delta=np.mean([np.sqrt(((d['prediction'][6,m]-d['truth'][m])**2).mean())-np.sqrt(((d['prediction'][4,m]-d['truth'][m])**2).mean()) for d in ds])
        loo.append(float(delta))
    output.append(dict(station=st,dates=daily,days_worse=sum(x['mean_seed_delta_sse']>0 for x in daily),
        leave_one_date_out_delta_rmse_range=[min(loo),max(loo)]))
with (P/'mapping_response_groups.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'MAPPING_DIAGNOSIS.json').write_text(json.dumps(dict(stations=output,new_fits=0,
    limits='Signs are descriptive: center COT cannot replace the full patch or impose a monotone GHI law; intervention is out of distribution.'),indent=2))
for x in output:print(x['station'],'days_worse',x['days_worse'],'of',len(x['dates']),'LOO',x['leave_one_date_out_delta_rmse_range'])
for r in rows:
    if r['station']=='Mizhuang':print(r)
