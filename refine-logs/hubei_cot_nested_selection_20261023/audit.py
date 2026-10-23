"""Audit inner-only teacher fitting, checkpoint choice, cold replay and FP64 scores."""
import hashlib,json,math
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
from model import Correction,DIRECT_COLUMNS,load_npz,response,inference

torch.set_num_threads(1);P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023';OLD=P.parent/'hubei_task_aligned_cot_20261023'
parts=json.loads((P/'INNER_PARTITIONS.json').read_text())['partitions'];prepared=json.loads((P/'prepared_attempt1/COMPLETE.json').read_text())
chosen=json.loads((P/'SELECTION.json').read_text());selection_sha=hashlib.sha256((P/'SELECTION.json').read_bytes()).hexdigest()
steps=[0,50,100,200,400,600];arms=['physical','task','direct_AGRI'];replaymax=0.;teachers=0;checkpoints=0;records=[]

def tensor(a,dtype=torch.float32):return torch.as_tensor(a,dtype=dtype)
def replay(c,arm,x,d,base,t,index,step):
    n=Correction(arm=='direct_AGRI');n.load_state_dict(c['models'][step]);n.eval()
    cols=DIRECT_COLUMNS if arm=='direct_AGRI' else list(range(308));z=tensor((x[index][:,cols]-c['mean'])/c['scale'])
    bt=tensor(base[index],torch.float64);av=tensor(d['available'][index],torch.bool)
    with torch.no_grad():
        if arm=='direct_AGRI':y=torch.where(av,torch.clamp_min(bt+1000*n(z).squeeze(-1),0),bt)
        else:
            ts=t['teacher_scale'] if 'teacher_scale' in t else t['scale']
            sens=tensor(response(t['coefficient'],ts,x[index,270:279].astype(bool)))
            original=tensor(x[index,279:288]);lo=torch.relu(original+n(z));y=inference(bt,av,sens,lo,original)
    return y.numpy()

for st in ['Niushou','Wangzhai','Mizhuang']:
    d=load_npz(Q/'inputs_attempt2'/f'{st}.npz');dates=np.array([str(t)[:8] for t in d['target_time']])
    for sj,seed in enumerate([42,43,44]):
        base=d['bases'][sj];x=np.column_stack([d['features'],base/d['clear']]);actual=x.copy();actual[:,279:297]=d['observed_COT_features']
        for outer in range(3):
            sums={a:np.zeros(6) for a in arms};count=0
            for inner in range(3):
                part=next(p for p in parts if p['outer_fold']==outer and p['inner_fold']==inner)
                file=P/'prepared_attempt1'/st/f'seed{seed}_outer{outer}_inner{inner}.npz';t=load_npz(file)
                receipt=next(r for r in prepared['records'] if (r['station'],r['seed'],r['outer_fold'],r['inner_fold'])==(st,seed,outer,inner))
                assert hashlib.sha256(file.read_bytes()).hexdigest()==receipt['sha256']
                tr=np.flatnonzero(np.isin(dates,part['train_dates'])&d['available']);ev=np.flatnonzero(np.isin(dates,part['inner_score_dates']))
                assert np.array_equal(t['train_index'],tr) and np.array_equal(t['score_index'],ev)
                forbidden={(datetime.strptime(day,'%Y%m%d')+timedelta(days=o)).strftime('%Y%m%d') for day in part['inner_score_dates']+part['outer_score_dates'] for o in [-1,0,1]}
                assert set(dates[tr]).isdisjoint(forbidden)
                assert np.allclose(t['teacher_mean'],actual[tr].mean(0),atol=1e-12,rtol=0)
                sd=actual[tr].std(0);sd=np.where(sd>1e-8,sd,1.)
                assert np.allclose(t['teacher_scale'],sd,atol=1e-12,rtol=0)
                assert abs(float(t['intercept'])-float(((d['truth'][tr]-base[tr])/1000).mean()))<1e-12
                az=(actual[tr]-t['teacher_mean'])/t['teacher_scale'];target=(d['truth'][tr]-base[tr])/1000-t['intercept']
                assert abs(az.T@(az@t['coefficient']-target)/len(tr)+.01*t['coefficient']).max()<1e-8
                teachers+=1;count+=len(ev)
                for arm in arms:
                    root=P/'inner_attempt1'/f'{st}_outer{outer}';predfile=root/f'seed{seed}_inner{inner}_{arm}_predictions.npz';v=load_npz(predfile)
                    assert hashlib.sha256(predfile.read_bytes()).hexdigest()==chosen['inner_prediction_sha256'][str(predfile.relative_to(P))]
                    assert np.array_equal(v['index'],ev) and np.array_equal(v['truth'],d['truth'][ev])
                    c=torch.load(root/f'seed{seed}_inner{inner}_{arm}_steps.pt',map_location='cpu')
                    assert c['updates']==600 and c['scheduler_last']['last_epoch']==600
                    cols=DIRECT_COLUMNS if arm=='direct_AGRI' else list(range(308))
                    expected=x[tr][:,cols];sd=expected.std(0);sd=np.where(sd>1e-8,sd,1.)
                    assert np.allclose(c['mean'],expected.mean(0),atol=1e-12,rtol=0) and np.allclose(c['scale'],sd,atol=1e-12,rtol=0)
                    order=np.random.default_rng(seed+1000*outer+10000*(inner+1)).integers(0,len(tr),(600,128))
                    assert hashlib.sha256(order.tobytes()).hexdigest()==c['sample_order_sha256']
                    assert all(float(s['step'])==600 for s in c['optimizer_last']['state'].values())
                    assert c['teacher_sha256']==receipt['sha256']
                    for j,step in enumerate(steps):
                        answer=replay(c,arm,x,d,base,t,ev,step)
                        error=float(abs(answer-v['prediction'][j]).max());replaymax=max(replaymax,error);assert error<.001
                        if step==0:assert np.array_equal(answer,base[ev])
                        sums[arm][j]+=math.fsum(float(e)**2 for e in v['prediction'][j]-d['truth'][ev])
                    checkpoints+=1
            for arm in arms:
                selection=next(c for c in chosen['selections'] if (c['station'],c['seed'],c['outer_fold'],c['arm'])==(st,seed,outer,arm))
                mse=sums[arm]/count;assert steps[int(np.argmin(mse))]==selection['step']
                assert np.allclose(mse,selection['inner_mse'],atol=1e-8,rtol=0)
                root=P/'outer_attempt1'/f'{st}_outer{outer}';complete=json.loads((root/'COMPLETE.json').read_text());assert complete['selection_sha256']==selection_sha
                c=torch.load(root/f'seed{seed}_inner-1_{arm}_steps.pt',map_location='cpu');old=torch.load(OLD/'training_attempt1'/st/f'seed{seed}_fold{outer}_{arm}_last.pt',map_location='cpu')
                assert all(torch.equal(v,old['model'][k]) for k,v in c['models'][600].items())
                v=load_npz(root/f'seed{seed}_{arm}_selected.npz');t=load_npz(Q/'probe_attempt1'/st/f'seed{seed}_observed_R_OFFLINE_fold{outer}.npz')
                answer=replay(c,arm,x,d,base,t,v['index'],selection['step']);error=float(abs(answer-v['prediction']).max());replaymax=max(replaymax,error);assert error<.001
                records.append(dict(station=st,seed=seed,outer=outer,arm=arm,selected_step=selection['step'],cold_prediction_error=error,step600_weight_difference=0))
                checkpoints+=1

days=sorted({day for p in parts for day in p['outer_score_dates']});stats=np.zeros((len(days),3,3,4,2))
for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
    for sj,seed in enumerate([42,43,44]):
        d=load_npz(P/'assembled_attempt1'/st/f'seed{seed}_OOF.npz')
        for di,day in enumerate(days):
            mask=d['date']==day
            for ai in range(4):stats[di,si,sj,ai]=[math.fsum(float(e)**2 for e in d['prediction'][ai,mask]-d['truth'][mask]),int(mask.sum())]
total=stats.sum(0);point=np.sqrt(total[...,0]/total[...,1]).mean((0,1));w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
flat=stats.reshape(len(days),-1,2);boot=np.sqrt((w@flat[:,:,0])/(w@flat[:,:,1])).reshape(5000,3,3,4).mean((1,2));result=json.loads((P/'RESULTS.json').read_text())['summary']['all'];names=['G1']+arms
pd=max(abs(point[i]-result['rmse'][name]) for i,name in enumerate(names));cd=0.
for j,k in [(2,1),(2,0),(2,3),(1,0),(3,0)]:cd=max(cd,float(abs(np.quantile(boot[:,j]-boot[:,k],[.025,.975])-result['comparisons'][f'{names[j]}_minus_{names[k]}']['ci95']).max()))
assert pd<1e-9 and cd<1e-9
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(state='PASS',inner_teachers=teachers,checkpoints=checkpoints,cold_prediction_max=replaymax,
    FP64_point_difference=float(pd),date_block_CI_difference=cd,selection_sha256=selection_sha,records=records,test_payloads_read=0),indent=2))
print('NESTED_AUDIT_PASS',teachers,checkpoints,replaymax,pd,cd)
