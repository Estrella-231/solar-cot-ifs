import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;result=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    folder=P/'sensitivity_attempt1'/st;check=json.loads((folder/'COMPLETE.json').read_text());assert check['state']=='PASS_FROZEN_INPUT_SENSITIVITY'
    for seed,receipt in zip([42,43,44],check['checks']):
        with np.load(folder/f'seed{seed}_sensitivity.npz') as z:d={k:z[k] for k in z.files}
        assert np.isfinite(d['derivative']).all() and (d['derivative'][d['available']==0]==0).all()
        assert np.all(d['fit']^d['select']) and len(set(zip(d['init_time'],d['target_time'],d['lead'])))==len(d['truth'])
        assert set(d['date'][d['fit']]).isdisjoint(set(d['date'][d['select']]))
        for scope in ['fit','select']:
            mask=d[scope]&(d['available']>0)
            for axis,name in enumerate(['center','all9_same_log_shift']):
                v=d['derivative'][mask,axis];active=abs(v)>1e-6
                result.append(dict(station=st,seed=seed,best_step=receipt['step'],scope=scope,axis=name,n=int(mask.sum()),
                    active_n=int(active.sum()),positive_fraction_all=float((v>1e-6).mean()),negative_fraction_all=float((v< -1e-6).mean()),
                    median=float(np.median(v)),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)),
                    mean_abs=float(abs(v).mean()),units='W m-2 per unit log1p(COT)',
                    GHI_bias=float((d['prediction'][mask]-d['truth'][mask]).mean())))
with (P/'sensitivity_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
for st in ['Niushou','Wangzhai','Mizhuang']:
    for axis in ['center','all9_same_log_shift']:
        for scope in ['fit','select']:
            r=[x for x in result if x['station']==st and x['axis']==axis and x['scope']==scope]
            print(st,axis,scope,[(x['seed'],x['best_step'],round(x['positive_fraction_all'],4),round(x['negative_fraction_all'],4),round(x['median'],5)) for x in r])
