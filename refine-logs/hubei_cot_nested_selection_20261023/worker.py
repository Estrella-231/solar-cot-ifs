"""Nested-date checkpoint training. Outer labels never choose checkpoints."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from model import Correction,DIRECT_COLUMNS,load_npz,response,losses,inference

P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023';OLD=P.parent/'hubei_task_aligned_cot_20261023'
STEPS=[0,50,100,200,400,600];ARMS=['physical','task','direct_AGRI']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['inner','outer','witness','profile'],required=True)
    ap.add_argument('--station',default='Niushou');ap.add_argument('--outer',type=int,default=0);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--tag',default='');a=ap.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(a.gpu);device=torch.device('cuda',a.gpu)
    torch.backends.cuda.matmul.allow_tf32=False
    assert json.loads((P/'prepared_attempt1/COMPLETE.json').read_text())['teachers']==81
    d=load_npz(Q/'inputs_attempt2'/f'{a.station}.npz');dates=np.array([str(t)[:8] for t in d['target_time']])
    folder={'inner':'inner_attempt1','outer':'outer_attempt1','witness':'witness_attempt1','profile':'profile_attempt1'}[a.mode]
    dest=P/folder/(a.tag or f'{a.station}_outer{a.outer}');dest.mkdir(parents=True,exist_ok=False)
    if a.mode=='outer':
        frozen=(P/'SELECTION.json').read_bytes();choices=json.loads(frozen)['selections'];selection_sha=hashlib.sha256(frozen).hexdigest()
    start=time.perf_counter();records=[]
    seeds=[42] if a.mode in ['witness','profile'] else [42,43,44]
    inners=range(3) if a.mode=='inner' else [0] if a.mode=='profile' else [-1]
    for seed in seeds:
        sj=[42,43,44].index(seed);base=d['bases'][sj];x=np.column_stack([d['features'],base/d['clear']])
        for inner in inners:
            if inner>=0:
                asset=P/'prepared_attempt1'/a.station/f'seed{seed}_outer{a.outer}_inner{inner}.npz';t=load_npz(asset)
                mean=t['mean'];scale=t['scale'];coef=t['coefficient'];ts=t['teacher_scale'];train=t['train_index'];score=t['score_index'];key=seed+1000*a.outer+10000*(inner+1)
            else:
                asset=Q/'probe_attempt1'/a.station/f'seed{seed}_observed_R_OFFLINE_fold{a.outer}.npz';t=load_npz(asset)
                f=load_npz(Q/'probe_attempt1'/a.station/f'seed{seed}_forecast_COT_fold{a.outer}.npz')
                mean=f['mean'];scale=f['scale'];coef=t['coefficient'];ts=t['scale'];train=np.flatnonzero(np.isin(dates,t['train_dates'])&d['available']);score=np.flatnonzero(np.isin(dates,t['test_dates']));key=seed+1000*a.outer
            def tensor(v,dtype=torch.float32):return torch.as_tensor(v,dtype=dtype,device=device)
            ix=tensor(train,torch.long);ev=tensor(score,torch.long);xn=tensor((x-mean)/scale)
            dm=x[train][:,DIRECT_COLUMNS].mean(0);ds=x[train][:,DIRECT_COLUMNS].std(0);ds=np.where(ds>1e-8,ds,1.)
            xd=tensor((x[:,DIRECT_COLUMNS]-dm)/ds);original=tensor(x[:,279:288]);ref=tensor(d['observed_COT_features'][:,:9])
            sens=tensor(response(coef,ts,x[:,270:279].astype(bool)));bt=tensor(base,torch.float64);yt=tensor(d['truth'],torch.float64);av=tensor(d['available'],torch.bool)
            order=np.random.default_rng(key).integers(0,len(train),(600,128));order_sha=hashlib.sha256(order.tobytes()).hexdigest()
            arms=['task'] if a.mode=='profile' else ARMS
            for arm in arms:
                torch.manual_seed(key);net=Correction(arm=='direct_AGRI').to(device)
                opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,600,eta_min=1e-5)
                snapshots={};predictions=[];cot_predictions=[];curves=[]
                def capture(step):
                    snapshots[step]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    if a.mode in ['witness','profile']:return
                    with torch.no_grad():
                        if arm=='direct_AGRI':
                            y=torch.where(av[ev],torch.clamp_min(bt[ev]+1000*net(xd[ev]).squeeze(-1),0),bt[ev])
                        else:
                            lo=torch.relu(original[ev]+net(xn[ev]));y=inference(bt[ev],av[ev],sens[ev],lo,original[ev]);cot_predictions.append(torch.expm1(lo).cpu().numpy())
                        assert torch.isfinite(y).all()
                        if step==0:assert torch.equal(y,bt[ev])
                        predictions.append(y.cpu().numpy())
                if a.mode!='profile':capture(0)
                for step in range(600):
                    idx=ix[tensor(order[step],torch.long)];opt.zero_grad(set_to_none=True)
                    if arm=='direct_AGRI':loss=(net(xd[idx]).squeeze(-1)-(yt[idx]-bt[idx])/1000).square().mean();lc=loss;lh=loss*0
                    else:
                        lo=torch.relu(original[idx]+net(xn[idx]));lc,lh=losses(lo,ref[idx],sens[idx]);loss=lc+(lh if arm=='task' else 0)
                    assert torch.isfinite(loss),(a.station,seed,a.outer,inner,arm,step)
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(grad)
                    opt.step();sched.step()
                    if step+1 in STEPS[1:] and a.mode!='profile':capture(step+1);curves.append([step+1,float(loss),float(lc),float(lh)])
                max_weight_diff=None
                if inner<0:
                    old=torch.load(OLD/'training_attempt1'/a.station/f'seed{seed}_fold{a.outer}_{arm}_last.pt',map_location='cpu')
                    assert old['sample_order_sha256']==order_sha
                    max_weight_diff=max(float((v.cpu()-old['model'][k]).abs().max()) for k,v in net.state_dict().items());assert max_weight_diff==0
                if a.mode!='profile':
                    file=dest/f'seed{seed}_inner{inner}_{arm}_steps.pt'
                    state=dict(models=snapshots,optimizer_last=opt.state_dict(),scheduler_last=sched.state_dict(),updates=600,steps=STEPS,
                        train_dates=t['train_dates'],mean=dm if arm=='direct_AGRI' else mean,scale=ds if arm=='direct_AGRI' else scale,
                        sample_order_sha256=order_sha,curves=curves,teacher_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
                    torch.save(state,file)
                    if a.mode!='witness':
                        predictions=np.stack(predictions)
                        if a.mode=='inner':
                            np.savez_compressed(dest/f'seed{seed}_inner{inner}_{arm}_predictions.npz',prediction=predictions,truth=d['truth'][score],index=score,steps=STEPS,date=dates[score])
                        else:
                            choice=next(c for c in choices if c['station']==a.station and c['seed']==seed and c['outer_fold']==a.outer and c['arm']==arm)
                            chosen=STEPS.index(choice['step'])
                            fields=dict(prediction=predictions[chosen],truth=d['truth'][score],base=base[score],index=score,step=choice['step'],
                                date=dates[score],init_time=d['init_time'][score],target_time=d['target_time'][score],lead=d['lead'][score],available=d['available'][score],source_role=d['source_role'][score])
                            if arm!='direct_AGRI':fields.update(cot_prediction=np.stack(cot_predictions)[chosen],cot_raw=np.expm1(x[score,279:288]),cot_reference=np.expm1(d['observed_COT_features'][score,:9]))
                            np.savez_compressed(dest/f'seed{seed}_{arm}_selected.npz',**fields)
                records.append(dict(seed=seed,outer=a.outer,inner=inner,arm=arm,updates=600,old_step600_weight_difference=max_weight_diff))
                print(a.mode,a.station,a.outer,seed,inner,arm,'600',flush=True)
    (dest/'COMPLETE.json').write_text(json.dumps(dict(mode=a.mode,station=a.station,outer=a.outer,records=records,seconds=time.perf_counter()-start,
        selection_sha256=selection_sha if a.mode=='outer' else None,test_payloads_read=0),indent=2))

if __name__=='__main__':main()
