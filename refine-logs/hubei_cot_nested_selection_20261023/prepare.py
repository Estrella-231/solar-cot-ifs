"""Fit inner teachers on INNER train dates only; no checkpoint/score selection."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from model import load_npz
P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023'
parts=json.loads((P/'INNER_PARTITIONS.json').read_text())['partitions']
O=P/'prepared_attempt1';O.mkdir(exist_ok=False);records=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    file=Q/'inputs_attempt2'/f'{st}.npz';d=load_npz(file)
    expected=next(r for r in json.loads((Q/'inputs_attempt2/COMPLETE.json').read_text())['stations'] if r['station']==st)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==expected['output_sha256']
    dates=np.array([str(t)[:8] for t in d['target_time']]);out=O/st;out.mkdir()
    for sj,seed in enumerate([42,43,44]):
        x=np.column_stack([d['features'],d['bases'][sj]/d['clear']]);actual=x.copy();actual[:,279:297]=d['observed_COT_features']
        for p in parts:
            train=np.isin(dates,p['train_dates'])&d['available'];score=np.isin(dates,p['inner_score_dates'])
            forbidden=set(p['outer_score_dates'])|set(p['inner_score_dates'])
            expanded={(datetime.strptime(day,'%Y%m%d')+timedelta(days=o)).strftime('%Y%m%d') for day in forbidden for o in [-1,0,1]}
            assert set(dates[train]).isdisjoint(expanded) and train.sum()>1000 and score.sum()>100
            mean=actual[train].mean(0);scale=actual[train].std(0);scale=np.where(scale>1e-8,scale,1.)
            z=(actual[train]-mean)/scale;y=(d['truth'][train]-d['bases'][sj,train])/1000;intercept=y.mean()
            coef=np.linalg.solve(z.T@z/len(z)+.01*np.eye(308),z.T@(y-intercept)/len(z))
            residual=float(abs(z.T@(z@coef-(y-intercept))/len(z)+.01*coef).max());assert residual<1e-8
            # Independent augmented least squares on the first teacher of each station.
            lstsq=None
            if seed==42 and p['outer_fold']==0 and p['inner_fold']==0:
                augmented=np.vstack([z/np.sqrt(len(z)),.1*np.eye(308)])
                target=np.r_[(y-intercept)/np.sqrt(len(z)),np.zeros(308)]
                other=np.linalg.lstsq(augmented,target,rcond=None)[0];lstsq=float(abs(other-coef).max());assert lstsq<1e-8
            fm=x[train].mean(0);fs=x[train].std(0);fs=np.where(fs>1e-8,fs,1.)
            target=out/f'seed{seed}_outer{p["outer_fold"]}_inner{p["inner_fold"]}.npz'
            np.savez_compressed(target,teacher_mean=mean,teacher_scale=scale,coefficient=coef,intercept=intercept,
                mean=fm,scale=fs,train_dates=np.array(sorted(set(dates[train]))),score_dates=np.array(sorted(set(dates[score]))),
                outer_score_dates=np.array(p['outer_score_dates']),train_index=np.flatnonzero(train),score_index=np.flatnonzero(score))
            records.append(dict(station=st,seed=seed,outer_fold=p['outer_fold'],inner_fold=p['inner_fold'],
                train_rows=int(train.sum()),score_rows=int(score.sum()),normal_error=residual,lstsq_difference=lstsq,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(O/'COMPLETE.json').write_text(json.dumps(dict(state='INNER_TRAIN_ONLY_TEACHERS_READY',teachers=len(records),records=records,
    inner_partitions_sha256=hashlib.sha256((P/'INNER_PARTITIONS.json').read_bytes()).hexdigest(),selected_steps=0,test_payloads_read=0),indent=2))
print('INNER_TEACHERS_READY',len(records))
