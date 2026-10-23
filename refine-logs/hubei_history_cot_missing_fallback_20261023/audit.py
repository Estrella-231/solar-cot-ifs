"""Second policy implementation and independent date/statistics accumulation."""
import json,math
from pathlib import Path
import numpy as np
from fallback import apply_history_missing_fallback
P=Path(__file__).resolve().parent;E=P.parent/'hubei_history_cot_confirmation_20261022'
result=json.loads((P/'RESULTS.json').read_text());checks=[];maximum=0.;ci_max=0.;nrows=0
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
# Same zero-valued history may be missing or a valid clear sky: policy uses QA only.
qa=np.zeros((4,8,1,3,3),dtype=bool);qa[1,0,0,1,1]=True;qa[2]=True
a=np.array([100.,100.,100.,100.]);b=np.array([90.,80.,70.,60.])
assert np.array_equal(apply_history_missing_fallback(a,b,qa),[100.,80.,70.,100.])
assert np.array_equal(apply_history_missing_fallback(np.stack([a,a]),np.stack([b,b]),qa),np.array([[100.,80.,70.,100.]]*2))
materials={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    d=load(P/'output_attempt1'/f'{st}.npz');base=[];cot=[]
    for seed in [42,43,44]:
        v=load(E/'predictions_attempt3'/f'{st}_seed{seed}_G0.npz');h=load(E/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz')
        for key in ['ghi','clear','init_time_BJT','target_time_BJT','lead_minutes','analysis']:assert np.array_equal(v[key],h[key])
        base.append(v['pred_ghi']);cot.append(h['pred_ghi'])
    base=np.array(base);cot=np.array(cot)
    flat=d['history_valid'].reshape(len(d['truth']),-1)
    missing=np.count_nonzero(flat,axis=1)==0;assert np.array_equal(missing,d['missing'])
    # In-place copy/assignment, independent of the np.where implementation.
    repaired=cot.copy();repaired[:,np.flatnonzero(missing)]=base[:,np.flatnonzero(missing)]
    rebuilt=np.stack([base,cot,repaired],1);assert np.array_equal(rebuilt,d['prediction'])
    assert np.array_equal(v['ghi'],d['truth'])
    materials[st]=d
    nrows+=int((d['analysis']=='primary_shared_dates').sum())
    checks.append(dict(station=st,all_prediction_rows=len(d['truth']),mask_replay_exact=True,prediction_replay_exact=True))
for group,reported in result['summary'].items():
    sites=list(reported['by_station']);days=reported['dates'];stat=np.zeros((len(days),len(sites),3,3,2))
    for si,st in enumerate(sites):
        d=materials[st];scope=d['analysis']=='primary_shared_dates';scope=scope if group=='primary' else ~scope
        for di,day in enumerate(days):
            m=scope&(d['date']==day)
            for sj in range(3):
                for ai in range(3):
                    err=d['prediction'][sj,ai,m]-d['truth'][m]
                    stat[di,si,sj,ai]=[float(np.dot(err,err)),int(m.sum())]
    sums=stat.sum(0);points=np.sqrt(sums[...,0]/sums[...,1]).mean((0,1));names=d['arms'].tolist()
    maximum=max(maximum,max(abs(float(points[i])-reported['rmse'][name]) for i,name in enumerate(names)))
    w=np.random.default_rng(20261022).multinomial(len(days),np.ones(len(days))/len(days),5000)
    ss=np.einsum('bd,dsja->bsja',w,stat[...,0]);count=np.einsum('bd,dsja->bsja',w,stat[...,1])
    boot=np.sqrt(ss/count).mean((1,2))
    for j,k in [(2,0),(2,1),(1,0)]:
        ci=np.percentile(boot[:,j]-boot[:,k],[2.5,97.5]);old=reported['comparisons'][f'{names[j]}_minus_{names[k]}']['ci95']
        ci_max=max(ci_max,float(abs(ci-old).max()))
assert maximum<1e-9 and ci_max<1e-9
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(state='PASS',checks=checks,primary_records=nrows,point_max_difference=maximum,
    date_CI_max_difference=ci_max,zero_COT_is_not_missing_test=True,partial_QA_does_not_fallback_test=True,new_model_fits=0,test_payloads_read=0),indent=2))
print('MISSING_POLICY_AUDIT_PASS',nrows,maximum,ci_max)
