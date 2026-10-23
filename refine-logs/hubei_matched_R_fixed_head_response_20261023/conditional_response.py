"""Fixed predefined thickness/date/lead decomposition; no tuning or fits."""
import csv, json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
rows=[]
for station in ['Niushou','Wangzhai','Mizhuang']:
    with np.load(P/'replay'/station/'physical_COT_sources.npz') as z:q={k:z[k] for k in z.files}
    for seed in [42,43,44]:
        with np.load(P/'replay'/station/f'seed{seed}_response.npz') as z:d={k:z[k] for k in z.files}
        valid=d['joint_available']>0; ref=q['observed_R'][:,4]; raw=q['forecast'][:,4]
        groups=[('reference_COT_0to1',(ref>=0)&(ref<1)),('reference_COT_1to10',(ref>=1)&(ref<10)),
                ('reference_COT_10to30',(ref>=10)&(ref<30)),('reference_COT_ge30',ref>=30)]
        groups += [(f'lead_{h}h',(d['lead']>(h-1)*60)&(d['lead']<=h*60)) for h in range(1,5)]
        groups += [(f'date_{day}',d['date']==day) for day in sorted(set(d['date']))]
        base=d['prediction'][4]; teacher=d['prediction'][6]; y=d['truth']
        for name,mask in groups:
            m=valid&mask
            if not m.any():continue
            shift=teacher[m]-base[m]; e=base[m]-y[m]; dc=ref[m]-raw[m]
            rows.append(dict(station=station,seed=seed,group=name,n=int(m.sum()),
                forecast_rmse=float(np.sqrt((e**2).mean())),
                reference_replaced_rmse=float(np.sqrt(((teacher[m]-y[m])**2).mean())),
                mean_ghi_shift=float(shift.mean()),mean_abs_ghi_shift=float(abs(shift).mean()),
                center_COT_mean_change=float(dc.mean()),delta_mse=float((2*e*shift+shift**2).mean()),
                change_energy=float((shift**2).mean()),interaction=float((2*e*shift).mean())))
with (P/'conditional_response.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for st in ['Niushou','Wangzhai','Mizhuang']:
    print(st)
    for group in ['reference_COT_0to1','reference_COT_1to10','reference_COT_10to30','reference_COT_ge30']:
        r=[v for v in rows if v['station']==st and v['group']==group]
        if r:print(group,r[0]['n'],'delta_rmse',np.mean([x['reference_replaced_rmse']-x['forecast_rmse'] for x in r]),'shift',np.mean([x['mean_ghi_shift'] for x in r]))
