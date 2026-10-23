import hashlib,json,math
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;O=P/'probe_attempt1';folds=json.loads((O/'DATE_FOLDS.json').read_text())['date_to_fold'];receipt=json.loads((O/'COMPLETE.json').read_text());audits={(r['station'],r['base_seed'],r['arm'],r['probe_fold']):r for r in receipt['audits']};checks=[];max_cold=0.;max_qr=0.
for st in ['Niushou','Wangzhai','Mizhuang']:
    with np.load(P/'inputs_attempt2'/f'{st}.npz') as z:d={k:z[k] for k in z.files}
    dates=np.array([str(t)[:8] for t in d['target_time']]);probe_fold=np.array([folds[t] for t in dates])
    for init,target,lead in zip(d['init_time'],d['target_time'],d['lead']):assert datetime.strptime(str(target),'%Y%m%d%H%M')==datetime.strptime(str(init),'%Y%m%d%H%M')+timedelta(minutes=int(lead)) and int(lead) in range(15,241,15)
    for sj,seed in enumerate([42,43,44]):
        with np.load(O/st/f'seed{seed}_OOF.npz') as z:saved={k:z[k] for k in z.files}
        assert np.array_equal(saved['truth'],d['truth']) and np.array_equal(saved['probe_fold'],probe_fold)
        for ai,arm in enumerate(['context','forecast_COT','observed_R_OFFLINE'],1):
            x=np.column_stack([d['features'],d['bases'][sj]/d['clear']])
            if arm=='context':x[:,279:297]=0
            if arm=='observed_R_OFFLINE':x[:,279:297]=d['observed_COT_features']
            for fold in range(3):
                file=O/st/f'seed{seed}_{arm}_fold{fold}.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==audits[st,seed,arm,fold]['model_sha256']
                with np.load(file) as z:fit={k:z[k] for k in z.files}
                held=[day for day,v in folds.items() if v==fold];excluded={ (datetime.strptime(day,'%Y%m%d')+timedelta(days=off)).strftime('%Y%m%d') for day in held for off in [-1,0,1] }
                train=(~np.isin(dates,list(excluded)))&d['available'];test=probe_fold==fold
                assert set(fit['train_dates'])==set(dates[train]) and set(fit['test_dates'])==set(dates[test]) and set(fit['train_dates']).isdisjoint(excluded)
                mean=x[train].mean(0);scale=x[train].std(0);scale=np.where(scale>1e-8,scale,1.)
                assert np.array_equal(mean,fit['mean']) and np.array_equal(scale,fit['scale'])
                q=np.maximum(0,d['bases'][sj,test]+1000*(float(fit['intercept'])+((x[test]-mean)/scale)@fit['coefficient']));q=np.where(d['available'][test],q,d['bases'][sj,test]);max_cold=max(max_cold,float(abs(q-saved['prediction'][ai,test]).max()))
                if seed==42 and arm=='context':
                    xx=(x[train]-mean)/scale;y=(d['truth'][train]-d['bases'][sj,train])/1000;yy=y-y.mean()
                    A=np.vstack([xx/np.sqrt(len(xx)),np.sqrt(.01)*np.eye(x.shape[1])]);b=np.r_[yy/np.sqrt(len(xx)),np.zeros(x.shape[1])]
                    qr=np.linalg.lstsq(A,b,rcond=None)[0];max_qr=max(max_qr,float(abs(qr-fit['coefficient']).max()))
                checks.append(dict(station=st,base_seed=seed,arm=arm,probe_fold=fold,train_rows=int(train.sum()),test_rows=int(test.sum())))
assert max_cold<1e-10 and max_qr<1e-8
(P/'MODEL_AUDIT.json').write_text(json.dumps(dict(state='PASS',models=len(checks),cold_replay_max=max_cold,independent_augmented_lstsq_max_coefficient_difference=max_qr,train_only_fold_statistics=True,target_time_and_purge_pass=True,test_payloads_read=0),indent=2));print('MODEL PASS',max_cold,max_qr)
# Independent date accumulators from predictions, including new cross-source branches.
for mode,folder,narms,comparison in [('probe',O,4,[(2,1),(3,1),(1,0),(2,0),(3,0)]),('cross_source',P/'cross_source_attempt1',6,[(4,1),(4,2),(4,0),(5,2),(3,2)])]:
    report=json.loads((P/'RESULTS.json' if mode=='probe' else folder/'RESULTS.json').read_text());pointmax=0.;cimax=0.
    for horizon in ['all','1h','2h','3h','4h']:
        expected=report['summary'][horizon];days=expected['dates'];stats=np.zeros((len(days),3,3,narms,2))
        for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
            for sj,seed in enumerate([42,43,44]):
                with np.load(folder/st/f'seed{seed}_OOF.npz') as z:d={k:z[k] for k in z.files}
                mask=np.ones(len(d['truth']),bool)
                if horizon!='all':h=int(horizon[0]);mask&=(d['lead']>(h-1)*60)&(d['lead']<=h*60)
                for di,day in enumerate(days):
                    mm=mask&(d['date']==day)
                    for ai in range(narms):stats[di,si,sj,ai]=[math.fsum(float(e)**2 for e in d['prediction'][ai,mm]-d['truth'][mm]),int(mm.sum())]
        total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]).mean((0,1));arms=list(expected['rmse']);pointmax=max(pointmax,max(abs(point[i]-expected['rmse'][arm]) for i,arm in enumerate(arms)))
        weights=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);boot=[]
        for w in weights:
            sums=np.tensordot(w,stats,axes=(0,0));boot.append(np.sqrt(sums[...,0]/sums[...,1]).mean((0,1)))
        boot=np.array(boot)
        for j,k in comparison:
            ci=np.quantile(boot[:,j]-boot[:,k],[.025,.975]);cimax=max(cimax,float(abs(ci-expected['comparisons'][f'{arms[j]}_minus_{arms[k]}']['ci95']).max()))
    assert pointmax<1e-10 and cimax<1e-10
    (P/f'{mode.upper()}_INDEPENDENT_SCORE.json').write_text(json.dumps(dict(state='PASS',max_point_difference=pointmax,max_ci_difference=cimax,test_payloads_read=0),indent=2));print(mode,'SCORES PASS',pointmax,cimax)
