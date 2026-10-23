"""Small CPU checks on all 27 date/source teachers; no formal training/scores."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from model import Correction, DIRECT_COLUMNS, response, losses, inference, load_npz

torch.set_num_threads(1)
P=Path(__file__).resolve().parent
Q=P.parent/'hubei_cot_conditional_information_20261023'
torch.manual_seed(42)
rows=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    d=load_npz(Q/'inputs_attempt2'/f'{st}.npz')
    dates=np.array([str(t)[:8] for t in d['target_time']])
    for sj, seed in enumerate([42,43,44]):
        x=np.column_stack([d['features'],d['bases'][sj]/d['clear']])
        for fold in range(3):
            m=load_npz(Q/'probe_attempt1'/st/f'seed{seed}_observed_R_OFFLINE_fold{fold}.npz')
            f=load_npz(Q/'probe_attempt1'/st/f'seed{seed}_forecast_COT_fold{fold}.npz')
            assert set(m['train_dates']).isdisjoint(m['test_dates'])
            assert np.array_equal(m['train_dates'],f['train_dates'])
            take=np.flatnonzero(np.isin(dates,m['train_dates']) & d['available'])[:32]
            xx=x[take].copy(); hv=xx[:,270:279].astype(bool)
            assert np.isfinite(xx).all()
            s=response(m['coefficient'],m['scale'],hv)
            delta=np.random.default_rng(seed+fold).normal(0,.01,(len(take),9))
            yy=xx.copy(); yy[:,279:288]+=delta; yy[:,288:297]+=delta*hv
            full=1000*(((yy-m['mean'])/m['scale'])@m['coefficient'] -
                       ((xx-m['mean'])/m['scale'])@m['coefficient'])
            linear=(s*delta).sum(-1)
            err=float(abs(full-linear).max()); assert err<1e-8
            fd=[]
            for i in range(9):
                a=xx.copy(); b=xx.copy(); h=1e-5
                a[:,279+i]+=h; b[:,279+i]-=h
                a[:,288+i]+=h*hv[:,i]; b[:,288+i]-=h*hv[:,i]
                finite=1000*((a-b)/m['scale']@m['coefficient'])/(2*h)
                fd.append(float(abs(finite-s[:,i]).max()))
            assert max(fd)<1e-6
            z=torch.as_tensor((xx-f['mean'])/f['scale'],dtype=torch.float64)
            original=torch.as_tensor(xx[:,279:288],dtype=torch.float64)
            reference=torch.as_tensor(d['observed_COT_features'][take,:9],dtype=torch.float64)
            sensitivity=torch.as_tensor(s,dtype=torch.float64)
            base=torch.as_tensor(d['bases'][sj,take],dtype=torch.float64)
            available=torch.ones(len(take),dtype=torch.bool)
            net=Correction().double(); direct=Correction(True).double()
            assert sum(p.numel() for p in net.parameters())==5369
            assert sum(p.numel() for p in direct.parameters())==5387
            log_out=torch.relu(original+net(z))
            assert torch.equal(log_out,original)
            assert torch.equal(inference(base,available,sensitivity,log_out,original),base)
            assert torch.equal(direct(z[:,DIRECT_COLUMNS]),torch.zeros(len(take),1,dtype=torch.float64))
            invalid=torch.zeros_like(available)
            assert torch.equal(inference(base,invalid,sensitivity,log_out+.1,original),base)
            lc,lh=losses(log_out,reference,sensitivity)
            (lc+lh).backward()
            grad=float(sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None))
            assert grad>0 and np.isfinite(grad)
            # Two updates, save optimizer/scheduler/model/RNG; resume the next exact update.
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,600,eta_min=1e-5)
            def step(n,o,c):
                o.zero_grad(); a,b=losses(torch.relu(original+n(z)),reference,sensitivity)
                loss=a+b; assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(n.parameters(),5)
                o.step(); c.step()
            step(net,opt,sch)
            snapshot=copy.deepcopy(dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict()))
            step(net,opt,sch)
            resumed=Correction().double(); resumed.load_state_dict(snapshot['model'])
            ro=torch.optim.AdamW(resumed.parameters(),lr=.001,weight_decay=.0001); ro.load_state_dict(snapshot['optimizer'])
            rs=torch.optim.lr_scheduler.CosineAnnealingLR(ro,600,eta_min=1e-5); rs.load_state_dict(snapshot['scheduler'])
            step(resumed,ro,rs)
            resume=max(float((a-b).abs().max()) for a,b in zip(net.parameters(),resumed.parameters()))
            assert resume==0
            rows.append(dict(station=st,seed=seed,fold=fold,linear_error=err,finite_difference_error=max(fd),gradient_l1=grad,resume_max=resume))
out=P/'ENGINEERING_CHECK.json'
assert not out.exists()
out.write_text(json.dumps(dict(state='CPU_ENGINEERING_PASS',teachers=len(rows),checks=rows,
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['model.py','check.py','PLAN.md']},
    formal_training_started=False,heldout_scores_computed=False,test_payloads_read=0),indent=2))
print('CPU_ENGINEERING_PASS',len(rows),flush=True)
