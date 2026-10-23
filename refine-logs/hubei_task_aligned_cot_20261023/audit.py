"""Cold checkpoint reconstruction and separate FP64/date-block score calculation."""
import hashlib,json,math
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
from model import Correction,DIRECT_COLUMNS,load_npz,response

torch.set_num_threads(1)
P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023'
sites=['Niushou','Wangzhai','Mizhuang'];days=sorted(json.loads((Q/'probe_attempt1/DATE_FOLDS.json').read_text())['date_to_fold'])
stats=np.zeros((len(days),3,3,4,2));records=[];maximum=0.
for si,st in enumerate(sites):
    d=load_npz(Q/'inputs_attempt2'/f'{st}.npz');dates=np.array([str(t)[:8] for t in d['target_time']])
    for sj,seed in enumerate([42,43,44]):
        saved=load_npz(P/'training_attempt1'/st/f'seed{seed}_OOF.npz')
        assert np.array_equal(saved['truth'],d['truth']) and np.array_equal(saved['target_time'],d['target_time'])
        assert np.array_equal(saved['source_role'],d['source_role'])
        assert np.array_equal(saved['prediction'][0],d['bases'][sj])
        for it,tt,l in zip(saved['init_time'],saved['target_time'],saved['lead']):
            assert datetime.strptime(str(it),'%Y%m%d%H%M')+timedelta(minutes=int(l))==datetime.strptime(str(tt),'%Y%m%d%H%M')
        x=np.column_stack([d['features'],d['bases'][sj]/d['clear']])
        for fold in range(3):
            t=load_npz(Q/'probe_attempt1'/st/f'seed{seed}_observed_R_OFFLINE_fold{fold}.npz')
            f=load_npz(Q/'probe_attempt1'/st/f'seed{seed}_forecast_COT_fold{fold}.npz')
            train=np.isin(dates,t['train_dates'])&d['available'];held=saved['fold']==fold
            exclude={(datetime.strptime(str(day),'%Y%m%d')+timedelta(days=off)).strftime('%Y%m%d') for day in t['test_dates'] for off in [-1,0,1]}
            assert set(dates[train]).isdisjoint(exclude)
            rng=np.random.default_rng(seed+1000*fold);order=rng.integers(0,int(train.sum()),(600,128));order_hash=hashlib.sha256(order.tobytes()).hexdigest()
            for ai,arm in enumerate(['physical','task','direct_AGRI'],1):
                file=P/'training_attempt1'/st/f'seed{seed}_fold{fold}_{arm}_last.pt'
                c=torch.load(file,map_location='cpu');assert c['step']==600 and c['sample_order_sha256']==order_hash
                assert c['scheduler']['last_epoch']==600
                assert all(float(v['step'])==600 for v in c['optimizer']['state'].values())
                assert len(c['train_dates'])==len(t['train_dates']) and np.array_equal(c['train_dates'],t['train_dates'])
                cols=DIRECT_COLUMNS if arm=='direct_AGRI' else list(range(308))
                mu=x[train][:,cols].mean(0);sd=x[train][:,cols].std(0);sd=np.where(sd>1e-8,sd,1.)
                assert np.allclose(c['mean'],mu,atol=1e-12,rtol=0) and np.allclose(c['scale'],sd,atol=1e-12,rtol=0)
                n=Correction(arm=='direct_AGRI');n.load_state_dict(c['model']);n.eval()
                with torch.no_grad():raw=n(torch.as_tensor((x[held][:,cols]-c['mean'])/c['scale'],dtype=torch.float32)).numpy()
                if arm=='direct_AGRI':answer=np.maximum(0,d['bases'][sj,held]+1000*raw[:,0])
                else:
                    original=x[held,279:288].astype('float32');lo=np.maximum(0,original+raw)
                    s=response(t['coefficient'],t['scale'],x[held,270:279].astype(bool)).astype('float32')
                    answer=np.maximum(0,d['bases'][sj,held]+(s*(lo-original)).sum(-1))
                    cd=float(np.abs(np.expm1(lo)-saved['cot_prediction'][ai-1,held]).max());assert cd<.001
                answer=np.where(d['available'][held],answer,d['bases'][sj,held])
                diff=float(abs(answer-saved['prediction'][ai,held]).max());maximum=max(maximum,diff);assert diff<.001
                records.append(dict(station=st,seed=seed,fold=fold,arm=arm,prediction_max_difference=diff,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        for di,day in enumerate(days):
            mask=dates==day
            for ai in range(4):
                err=saved['prediction'][ai,mask]-d['truth'][mask]
                stats[di,si,sj,ai]=[math.fsum(float(e)**2 for e in err),len(err)]
total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]).mean((0,1))
w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
flat=stats.reshape(len(days),-1,2);bs=np.sqrt((w@flat[:,:,0])/(w@flat[:,:,1])).reshape(5000,3,3,4).mean((1,2))
primary=json.loads((P/'RESULTS.json').read_text())['summary']['all'];names=['G1','physical','task','direct_AGRI']
pd=max(abs(float(point[i])-primary['rmse'][name]) for i,name in enumerate(names));cierr=0.
for j,k in [(2,1),(2,0),(2,3),(1,0),(3,0)]:
    ci=np.percentile(bs[:,j]-bs[:,k],[2.5,97.5]);old=primary['comparisons'][f'{names[j]}_minus_{names[k]}']['ci95'];cierr=max(cierr,float(abs(ci-old).max()))
assert pd<1e-9 and cierr<1e-9
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(state='PASS',checkpoints=len(records),records=records,
    cpu_cold_prediction_max=maximum,FP64_point_difference=pd,date_block_CI_difference=cierr,test_payloads_read=0),indent=2))
print('AUDIT_PASS',len(records),maximum,pd,cierr)
