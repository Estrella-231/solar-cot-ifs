"""Frozen correction error decomposition; no fitting or data-dependent cutpoints."""
import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
rows=[]; daily=[]; witnesses=[]
def metric(base,corrected,y):
    e=base-y;d=corrected-base;err=corrected-y
    identity=float(np.max(abs(err*err-e*e-(2*e*d+d*d))))
    assert identity<1e-8
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(err*err))),bias=float(err.mean()),mae=float(abs(err).mean()),
        base_rmse=float(np.sqrt(np.mean(e*e))),base_bias=float(e.mean()),mean_adjustment=float(d.mean()),
        abs_adjustment=float(abs(d).mean()),interaction=float((2*e*d).mean()),adjustment_energy=float((d*d).mean()),
        delta_mse=float((err*err-e*e).mean()),direction_correct_fraction=float((e*d<0).mean()),
        reduced_squared_error_fraction=float((err*err<e*e).mean()),identity_error=identity)
for st in ['Niushou','Wangzhai','Mizhuang']:
    for seed in [42,43,44]:
        f=P/'replay'/st/f'seed{seed}_replay.npz'
        with np.load(f) as z:data={k:z[k] for k in z.files}
        pred=data['prediction'].astype(np.float64);truth=data['truth'].astype(np.float64);arms=data['arms'].tolist()
        assert arms==['A','H','F','FS','M','V']
        for role in ['C_fit','C_select']:
            rm=data['roles']==role
            groups=[('all',np.ones(len(truth),bool)),('GHI_lt200',truth<200),('GHI_200to500',(truth>=200)&(truth<500)),
                ('GHI_500to800',(truth>=500)&(truth<800)),('GHI_ge800',truth>=800),
                ('future_cot_missing',~data['future_cot_valid'].any(1)),('future_cot_valid',data['future_cot_valid'].any(1))]
            cot=data['future_cot_center'];cv=data['future_cot_valid'][:,4]
            groups += [(f'cot_{lo}to{hi}',cv&(cot>=lo)&(cot<hi)) for lo,hi in [(0,3),(3,10),(10,30),(30,101)]]
            groups += [(f'hour_{h}',(data['lead']>(h-1)*60)&(data['lead']<=h*60)) for h in range(1,5)]
            for group,mask in groups:
                mask=mask&rm
                if not mask.any():continue
                for j,arm in enumerate(arms):
                    rows.append(dict(station=st,seed=seed,role=role,group=group,arm=arm,**metric(pred[0,mask],pred[j,mask],truth[mask])))
            for date in sorted(set(data['date'][rm])):
                m=rm&(data['date']==date)
                for j,arm in enumerate(arms):daily.append(dict(station=st,seed=seed,role=role,date=date,arm=arm,**metric(pred[0,m],pred[j,m],truth[m])))
        prior=P.parent/'screen_attempt2'/st/f'seed{seed}_Cselect.npz'
        with np.load(prior) as z:
            diff=float(abs(z['prediction']-pred[:,data['roles']=='C_select']).max());assert diff<1e-8
        witnesses.append(dict(station=st,seed=seed,old_prediction_max_abs_difference=diff))
summary={}
for role in ['C_fit','C_select']:
    summary[role]={}
    for arm in ['A','H','F','FS','M','V']:
        rr=[r for r in rows if r['role']==role and r['group']=='all' and r['arm']==arm]
        summary[role][arm]={k:math.fsum(r[k] for r in rr)/len(rr) for k in ['rmse','bias','base_bias','mean_adjustment','abs_adjustment','interaction','adjustment_energy','delta_mse','direction_correct_fraction','reduced_squared_error_fraction']}
for name,rr in [('groups',rows),('daily',daily)]:
    out=P/(name+'.csv');assert not out.exists()
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
report=dict(state='FROZEN_COEFFICIENT_ATTRIBUTION_COMPLETE',summary=summary,witnesses=witnesses,
    max_identity_error=max(r['identity_error'] for r in rows),new_fits=0,new_selection=0,official_val_test_new_period_labels_read=0)
(P/'ANALYSIS.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
