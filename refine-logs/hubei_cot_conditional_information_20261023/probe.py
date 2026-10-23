"""One fixed ridge per date fold, all statistics fitted on that fold's train rows."""
import csv,hashlib,json,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;start=time.perf_counter();inp=P/'inputs_attempt2';meta=json.loads((inp/'COMPLETE.json').read_text());assert meta['state']=='CFIT_ONLY_PROBE_INPUTS_READY'
sites=['Niushou','Wangzhai','Mizhuang'];sources={}
for st in sites:
    file=inp/f'{st}.npz';receipt=next(r for r in meta['stations'] if r['station']==st);assert hashlib.sha256(file.read_bytes()).hexdigest()==receipt['output_sha256']
    with np.load(file) as z:sources[st]={k:z[k] for k in z.files}
all_dates=sorted({str(t)[:8] for d in sources.values() for t in d['target_time']});mapping={};blocks=[]
for month in sorted({d[:6] for d in all_dates}):
    days=[d for d in all_dates if d.startswith(month)];components=[]
    for day in days:
        if not components or (datetime.strptime(day,'%Y%m%d')-datetime.strptime(components[-1][-1],'%Y%m%d')).days>1:components.append([])
        components[-1].append(day)
    for j,component in enumerate(components):
        fold=j%3;blocks.append(dict(month=month,dates=component,probe_fold=fold));mapping.update({d:fold for d in component})
out=P/'probe_attempt1';out.mkdir(exist_ok=False);(out/'DATE_FOLDS.json').write_text(json.dumps(dict(date_to_fold=mapping,components=blocks,purge_days=1),indent=2));audits=[]
arms=['G1','context','forecast_COT','observed_R_OFFLINE']
for st in sites:
    d=sources[st];dates=np.array([str(t)[:8] for t in d['target_time']]);folds=np.array([mapping[t] for t in dates]);available=d['available'];dest=out/st;dest.mkdir()
    assert set(folds)=={0,1,2} and len(set(zip(d['init_time'],d['target_time'],d['lead'])))==len(dates)
    for sj,seed in enumerate([42,43,44]):
        base=d['bases'][sj];truth=d['truth'];assert np.all(d['clear']>0)
        x0=np.column_stack([d['features'],base/d['clear']]);pred=np.full((4,len(dates)),np.nan);pred[0]=base
        for ai,arm in enumerate(arms[1:],1):
            x=x0.copy()
            if arm=='context':x[:,279:297]=0
            if arm=='observed_R_OFFLINE':x[:,279:297]=d['observed_COT_features']
            for fold in range(3):
                held=sorted(day for day,v in mapping.items() if v==fold);exclude={ (datetime.strptime(day,'%Y%m%d')+timedelta(days=off)).strftime('%Y%m%d') for day in held for off in [-1,0,1] }
                train=(~np.isin(dates,list(exclude)))&available;test=folds==fold
                train_dates=sorted(set(dates[train]));assert len(train_dates)>=10 and train.sum()>1000 and test.sum()>100
                assert set(train_dates).isdisjoint(exclude)
                mu=x[train].mean(0);scale=x[train].std(0);scale=np.where(scale>1e-8,scale,1.)
                xx=(x[train]-mu)/scale;y=(truth[train]-base[train])/1000;intercept=float(y.mean());centered=y-intercept
                coefficient=np.linalg.solve(xx.T@xx/len(xx)+.01*np.eye(x.shape[1]),xx.T@centered/len(xx))
                normal_error=float(abs(xx.T@(xx@coefficient-centered)/len(xx)+.01*coefficient).max());assert normal_error<1e-8
                q=np.maximum(0,base[test]+1000*(intercept+(x[test]-mu)/scale@coefficient));q=np.where(available[test],q,base[test]);pred[ai,test]=q
                file=dest/f'seed{seed}_{arm}_fold{fold}.npz';np.savez_compressed(file,mean=mu,scale=scale,coefficient=coefficient,intercept=intercept,train_dates=np.array(train_dates),test_dates=np.array(sorted(set(dates[test]))),ridge_alpha=.01)
                audits.append(dict(station=st,base_seed=seed,arm=arm,probe_fold=fold,train_rows=int(train.sum()),train_dates=len(train_dates),test_rows=int(test.sum()),normal_equation_max_error=normal_error,constant_columns=int((xx.std(0)<1e-10).sum()),model_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        assert np.isfinite(pred).all() and np.array_equal(pred[:,~available],np.broadcast_to(base[~available],(4,int((~available).sum()))))
        np.savez_compressed(dest/f'seed{seed}_OOF.npz',prediction=pred,truth=truth,clear=d['clear'],arms=arms,date=dates,probe_fold=folds,available=available,init_time=d['init_time'],target_time=d['target_time'],lead=d['lead'],upstream_source_role=d['source_role'])
        print(st,seed,'OOF complete',flush=True)
(out/'COMPLETE.json').write_text(json.dumps(dict(state='FIXED_DATE_OOF_RIDGE_COMPLETE',fold_models=len(audits),audits=audits,seconds=time.perf_counter()-start,new_GPU_runs=0,C_select_scored_rows=0,test_payloads_read=0,claim='Train-only fixed linear feature diagnostic; observed-R arm offline only'),indent=2));print('COMPLETE',len(audits),time.perf_counter()-start)
