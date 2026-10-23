import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;rows=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    with np.load(P/'inputs_attempt2'/f'{st}.npz') as z:raw={k:z[k] for k in ['observed_COT_features','available']}
    cot=np.expm1(raw['observed_COT_features'][:,4]);av=raw['available']
    for seed in [42,43,44]:
        with np.load(P/'cross_source_attempt1'/st/f'seed{seed}_OOF.npz') as z:d={k:z[k] for k in z.files}
        true=d['prediction'][3];forecast=d['prediction'][4];e=true-d['truth'];shift=forecast-true
        net=(forecast-d['truth'])**2-e**2;identity=2*e*shift+shift**2;assert float(abs(net-identity).max())<1e-7
        groups=[('all',np.ones(len(e),bool)),('available',av),('unavailable',~av)]
        groups += [(f'lead_{h}h',(d['lead']>(h-1)*60)&(d['lead']<=h*60)) for h in range(1,5)]
        groups += [('reference_COT_0to1',av&(cot<1)),('reference_COT_1to10',av&(cot>=1)&(cot<10)),('reference_COT_10to30',av&(cot>=10)&(cot<30)),('reference_COT_ge30',av&(cot>=30))]
        for name,m in groups:
            if not m.any():continue
            rows.append(dict(station=st,base_seed=seed,group=name,n=int(m.sum()),reference_input_rmse=float(np.sqrt((e[m]**2).mean())),forecast_input_rmse=float(np.sqrt(((forecast[m]-d['truth'][m])**2).mean())),mean_source_shift=float(shift[m].mean()),mean_abs_source_shift=float(abs(shift[m]).mean()),delta_mse=float(net[m].mean()),interaction=float((2*e[m]*shift[m]).mean()),shift_energy=float((shift[m]**2).mean())))
with (P/'source_penalty.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for st in ['Niushou','Wangzhai','Mizhuang']:
    r=[x for x in rows if x['station']==st and x['group']=='all'];print(st,{k:float(np.mean([x[k] for x in r])) for k in ['mean_source_shift','mean_abs_source_shift','delta_mse','interaction','shift_energy']})
