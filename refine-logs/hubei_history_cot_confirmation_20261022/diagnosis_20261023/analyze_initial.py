import numpy as np,json,csv
from pathlib import Path
p=Path('solar-cot-ifs/refine-logs/hubei_history_cot_confirmation_20261022'); out=p/'diagnosis_20261023'; byinit={}; result={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    aa=[];bb=[]
    for seed in [42,43,44]:
        with np.load(p/'predictions_attempt3'/f'{st}_seed{seed}_G0.npz') as z: a={k:z[k] for k in z.files}
        with np.load(p/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz') as z: b={k:z[k] for k in z.files}
        aa.append(a['pred_ghi']);bb.append(b['pred_ghi'])
    x=np.array(aa);y=np.array(bb);obs=a['ghi']; m=a['analysis']=='primary_shared_dates'; e=x-obs; d=y-x; ds=(y-obs)**2-e**2
    for t in sorted(set(a['init_time_BJT'][m])):
        ix=(a['init_time_BJT']==t)&m
        if ix.sum()>=12: byinit.setdefault(t,{})[st]=float(ds[:,ix].mean())
    result[st]=dict(n=int(m.sum()),rmse0=float(np.sqrt((e[:,m]**2).mean(1)).mean()),rmse1=float(np.sqrt(((y[:,m]-obs[m])**2).mean(1)).mean()),mean_adjustment=float(d[:,m].mean()),mean_abs_adjustment=float(abs(d[:,m]).mean()),fraction_improved=float((ds[:,m]<0).mean()),baseline_bias=float(e[:,m].mean()),new_bias=float((y[:,m]-obs[m]).mean()),interaction=float((2*e[:,m]*d[:,m]).mean()),correction_energy=float((d[:,m]**2).mean()),delta_mse=float(ds[:,m].mean()),seed_delta_rmse=(np.sqrt(((y[:,m]-obs[m])**2).mean(1))-np.sqrt((e[:,m]**2).mean(1))).tolist())
rows=[]
for t,ss in byinit.items():
    if len(ss)==3:
        rows.append(dict(init=t,**ss,contrast=ss['Wangzhai']+ss['Mizhuang']-ss['Niushou']))
rows.sort(key=lambda r:-r['contrast']); choices=[]
for r in rows:
    if r['Niushou']<0 and r['Wangzhai']>0 and r['Mizhuang']>0 and r['init'][:10] not in [x['init'][:10] for x in choices]:choices.append(r)
    if len(choices)==3:break
(out/'PRELIMINARY.json').write_text(json.dumps(dict(stations=result,candidate_cases=choices),indent=2))
with (out/'all_common_initializations.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(stations=result,candidate_cases=choices),indent=2))

