"""Read saved fixed-head C-role predictions; descriptive decomposition only."""
import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;records=[];daily=[];fallback=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    for seed in [42,43,44]:
        with np.load(P/'replay'/st/f'seed{seed}_replay.npz') as z:d={k:z[k] for k in z.files}
        y=d['truth'].astype('float64');pred=d['prediction'].astype('float64');arms=d['arms'].tolist();base=pred[0]
        valid=d['solar_available'].astype(bool)
        for j in [5,6]:assert np.array_equal(pred[j,~valid],base[~valid])
        for role in ['C_fit','C_select']:
            rm=d['roles']==role
            fallback.append(dict(station=st,seed=seed,role=role,n=int(rm.sum()),solar_invalid=int((rm&~valid).sum()),old_available_invalid=int((rm&~valid&d['original_available'].astype(bool)).sum())))
            groups=[('all',np.ones(len(y),bool)),('solar_valid',valid),('solar_invalid',~valid),('GHI_ge800',y>=800),('base_underestimate',base<y),('base_overestimate',base>y)]
            groups += [(f'hour_{h}',(d['lead']>(h-1)*60)&(d['lead']<=h*60)) for h in range(1,5)]
            def metrics(m,j):
                e=base[m]-y[m];delta=pred[j,m]-base[m];q=e+delta
                identity=float(abs(q*q-e*e-(2*e*delta+delta*delta)).max());assert identity<1e-8
                return dict(n=int(m.sum()),rmse=float(np.sqrt(np.mean(q*q))),bias=float(q.mean()),mae=float(abs(q).mean()),
                    mean_adjustment=float(delta.mean()),mean_abs_adjustment=float(abs(delta).mean()),interaction=float(np.mean(2*e*delta)),adjustment_energy=float(np.mean(delta*delta)),
                    delta_mse=float(np.mean(q*q-e*e)),direction_correct=float(np.mean(e*delta<0)),squared_error_improved=float(np.mean(q*q<e*e)),identity_error=identity)
            for group,m in groups:
                m=m&rm
                if not m.any():continue
                for j,arm in enumerate(arms):records.append(dict(station=st,seed=seed,role=role,group=group,arm=arm,**metrics(m,j)))
            for day in sorted(set(d['date'][rm])):
                m=rm&(d['date']==day)
                for j,arm in enumerate(arms):daily.append(dict(station=st,seed=seed,role=role,date=day,arm=arm,**metrics(m,j)))
summary={}
for role in ['C_fit','C_select']:
    summary[role]={}
    for arm in arms:
        rr=[r for r in records if r['role']==role and r['group']=='all' and r['arm']==arm]
        summary[role][arm]={k:math.fsum(r[k] for r in rr)/len(rr) for k in ['rmse','bias','mean_adjustment','mean_abs_adjustment','interaction','adjustment_energy','delta_mse','direction_correct','squared_error_improved']}
for name,rr in [('groups',records),('daily',daily),('fallback',fallback)]:
    f=P/(name+'.csv');assert not f.exists()
    with f.open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
report=dict(state='NEURAL_C_ROLE_DECOMPOSITION_COMPLETE',summary=summary,new_fits=0,new_model_selection=0,official_val_new_period_test_payloads_read=0)
(P/'ANALYSIS.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
