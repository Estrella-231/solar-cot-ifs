"""Posthoc target adequacy audit; no fitting or parameter selection."""
import csv,json,math
from pathlib import Path
import numpy as np
import torch
from model import Correction,DIRECT_COLUMNS,load_npz,response

torch.set_num_threads(1);P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023'
out=P/'target_diagnostic_attempt1';out.mkdir(exist_ok=False);rows=[];oofrecords=[]

def summarize(base,truth,delta):
    e=base-truth;ans=np.maximum(0,base+delta);actual=ans-base;err=ans-truth
    cross=math.fsum(float(a)*float(b)*2 for a,b in zip(e,actual))/len(e)
    square=math.fsum(float(v)**2 for v in actual)/len(e)
    diff=math.fsum(float(a)**2-float(b)**2 for a,b in zip(err,e))/len(e)
    assert abs(diff-cross-square)<1e-7
    nonzero=(np.abs(actual)>1e-8)&(np.abs(e)>1e-8)
    def cosine(a,b):
        den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den else None
    den=float(delta@delta)
    return dict(n=len(e),base_rmse=math.sqrt(math.fsum(float(v)**2 for v in e)/len(e)),
        rmse=math.sqrt(math.fsum(float(v)**2 for v in err)/len(e)),mae=float(abs(err).mean()),bias=float(err.mean()),
        mean_shift=float(actual.mean()),mean_abs_shift=float(abs(actual).mean()),cross_2ed=cross,delta_square=square,delta_MSE=diff,
        algebra_error=abs(diff-cross-square),nonzero_rows=int(nonzero.sum()),
        correction_direction_fraction=float((actual[nonzero]*e[nonzero]<0).mean()) if nonzero.any() else None,
        cosine_to_needed_residual=cosine(delta,-e),diagnostic_projection_coefficient=float(delta@(-e)/den) if den else None)

for st in ['Niushou','Wangzhai','Mizhuang']:
    d=load_npz(Q/'inputs_attempt2'/f'{st}.npz');dates=np.array([str(t)[:8] for t in d['target_time']]);av=d['available']
    for sj,seed in enumerate([42,43,44]):
        saved=load_npz(P/'training_attempt1'/st/f'seed{seed}_OOF.npz');base=d['bases'][sj]
        x=np.column_stack([d['features'],base/d['clear']]);reference=d['observed_COT_features'][:,:9];original=x[:,279:288]
        allpred=np.full((5,len(base)),np.nan);allpred[0]=base
        for fold in range(3):
            teacher=load_npz(Q/'probe_attempt1'/st/f'seed{seed}_observed_R_OFFLINE_fold{fold}.npz')
            sens=response(teacher['coefficient'],teacher['scale'],x[:,270:279].astype(bool))
            oracle=np.where(av,(sens*(reference-original)).sum(-1),0.)
            masks={'fit':np.isin(dates,teacher['train_dates'])&av,'heldout':saved['fold']==fold}
            deltas={'reference_innovation_OFFLINE':oracle}
            logoutputs={}
            for arm in ['physical','task','direct_AGRI']:
                c=torch.load(P/'training_attempt1'/st/f'seed{seed}_fold{fold}_{arm}_last.pt',map_location='cpu')
                net=Correction(arm=='direct_AGRI');net.load_state_dict(c['model']);net.eval()
                cols=DIRECT_COLUMNS if arm=='direct_AGRI' else list(range(308))
                with torch.no_grad():raw=net(torch.as_tensor((x[:,cols]-c['mean'])/c['scale'],dtype=torch.float32)).numpy().astype('float64')
                if arm=='direct_AGRI':delta=1000*raw[:,0]
                else:
                    logs=np.maximum(0,original+raw);logoutputs[arm]=logs
                    delta=(sens*(logs-original)).sum(-1)
                deltas[arm]=np.where(av,delta,0.)
            for domain,mask in masks.items():
                for arm,delta in deltas.items():
                    record=dict(station=st,seed=seed,fold=fold,domain=domain,arm=arm,**summarize(base[mask],d['truth'][mask],delta[mask]))
                    if arm in logoutputs:
                        valid=mask&av;logs=logoutputs[arm][valid];ref=reference[valid]
                        record['COT_reference_MSE']=float((((np.expm1(logs)-np.expm1(ref))/100)**2).mean())
                        record['teacher_response_MSE']=float((((sens[valid]*(logs-ref)).sum(-1)/1000)**2).mean())
                    rows.append(record)
            held=masks['heldout']
            for ai,arm in enumerate(deltas,1):allpred[ai,held]=np.maximum(0,base[held]+deltas[arm][held])
        assert np.isfinite(allpred).all()
        assert np.abs(allpred[2:]-saved['prediction'][1:]).max()<.001
        for ai,arm in enumerate(['G1','reference_innovation_OFFLINE','physical','task','direct_AGRI']):
            oofrecords.append(dict(station=st,seed=seed,arm=arm,**summarize(base,d['truth'],allpred[ai]-base)))
        np.savez_compressed(out/f'{st}_seed{seed}_OOF.npz',prediction=allpred,truth=d['truth'],date=dates,
            init_time=d['init_time'],target_time=d['target_time'],lead=d['lead'],available=av,
            arms=['G1','reference_innovation_OFFLINE','physical','task','direct_AGRI'])
fields=sorted({k for r in rows for k in r})
with (out/'fold_domain.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (out/'OOF.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(oofrecords[0]));w.writeheader();w.writerows(oofrecords)
summary={}
for arm in ['reference_innovation_OFFLINE','physical','task','direct_AGRI']:
    summary[arm]={}
    for domain in ['fit','heldout']:
        rr=[r for r in rows if r['arm']==arm and r['domain']==domain]
        summary[arm][domain]={k:float(np.mean([r[k] for r in rr])) for k in ['rmse','base_rmse','delta_MSE','cross_2ed','delta_square','mean_abs_shift','correction_direction_fraction']}
        if arm in ['physical','task']:
            summary[arm][domain].update({k:float(np.mean([r[k] for r in rr])) for k in ['COT_reference_MSE','teacher_response_MSE']})
    rr=[r for r in oofrecords if r['arm']==arm]
    summary[arm]['OOF']={k:float(np.mean([r[k] for r in rr])) for k in ['rmse','base_rmse','delta_MSE','cross_2ed','delta_square','mean_abs_shift','correction_direction_fraction']}
(out/'COMPLETE.json').write_text(json.dumps(dict(state='POSTHOC_TARGET_DIAGNOSTIC_COMPLETE',summary=summary,
    fit_rows_repeated_across_outer_models=True,OOF_rows=32838,new_model_fits=0,test_payloads_read=0),indent=2))
print(json.dumps(summary,indent=2))
