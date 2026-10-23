import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;data={};dates=set()
for st in ['Niushou','Wangzhai','Mizhuang']:
    for seed in [42,43,44]:
        with np.load(P/'cross_source_attempt1'/st/f'seed{seed}_OOF.npz') as z:d={k:z[k] for k in z.files}
        data[st,seed]=d;dates.update(d['date'])
out={}
for name,j,k in [('forecast_COT_vs_context',2,1),('actual_COT_OFFLINE_vs_G1',3,0),('actual_trained_forecast_input_vs_G1',4,0)]:
    differences=[];days=[]
    for omit in sorted(dates):
        v=[]
        for d in data.values():
            mask=d['date']!=omit;y=d['truth'][mask];v.append(np.sqrt(((d['prediction'][j,mask]-y)**2).mean())-np.sqrt(((d['prediction'][k,mask]-y)**2).mean()))
        delta=float(np.mean(v));differences.append(delta);days.append(dict(omitted_date=str(omit),delta_rmse=delta))
    out[name]=dict(leave_one_date_out_range=[min(differences),max(differences)],details=days)
(P/'DATE_SENSITIVITY.json').write_text(json.dumps(out,indent=2));print({k:v['leave_one_date_out_range'] for k,v in out.items()})
