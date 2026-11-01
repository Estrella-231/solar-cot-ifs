"""Paired forecast-only GHI field-of-view pilot; old-contract diagnostic.

R is evaluated on full256 before cropping in the immutable source bank.
No oracle arrays, regional branch, observed GHI input, or test payloads.
All sizes use the same 64 canvas and parameter count. Crop before convolution.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WINDOWS = (1, 3, 5, 16, 32, 64)
SEEDS = (42, 43, 44)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def seed(value):
    random.seed(value); np.random.seed(value)
    torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bounds(size):
    lo = 32 - size//2
    return lo, lo+size


def crop_pad(x, size):
    lo, hi = bounds(size)
    return F.pad(x[..., lo:hi, lo:hi], (lo, 64-hi, lo, 64-hi))


class Head(nn.Module):
    """Chronologically stacked COT prefix, fixed spatial conv and grid readout."""
    def __init__(self):
        super().__init__()
        # 16 forecast AGRI/geometry + 24 ordered COT + spatial support.
        self.spatial = nn.Sequential(nn.Conv2d(41,16,3,2,1), nn.SiLU(),
                                     nn.Conv2d(16,16,3,2,1), nn.SiLU())
        self.grid = nn.Sequential(nn.Flatten(), nn.Linear(16*16*16,64), nn.SiLU())
        self.station = nn.Embedding(2,8)
        self.lead = nn.Embedding(16,8)
        self.read = nn.Sequential(nn.Linear(80,128), nn.SiLU(), nn.Linear(128,1), nn.Softplus())

    def forward(self, image, cot, station, lead, size, use_cot):
        assert image.shape[1:] == (16,64,64) and cot.shape[1:] == (24,64,64)
        image = crop_pad(image, size)
        prefix = torch.arange(24,device=cot.device)[None] <= 8+lead[:,None]
        cot = crop_pad(cot, size) * prefix[:,:,None,None]
        if not use_cot: cot = torch.zeros_like(cot)
        support = crop_pad(torch.ones_like(image[:,:1]), size)
        z = self.grid(self.spatial(torch.cat((image,cot,support),1)))
        return self.read(torch.cat((z,self.station(station),self.lead(lead)),1)).squeeze(1)


def witness(device):
    seed(42)
    model = Head().to(device)
    image = torch.randn(2,16,64,64,device=device,requires_grad=True)
    cot = torch.randn(2,24,64,64,device=device,requires_grad=True)
    st = torch.tensor([0,1],device=device); lead = torch.tensor([0,7],device=device)
    checks = []
    for size in WINDOWS:
        pred = model(image,cot,st,lead,size,True)
        gimage,gcot = torch.autograd.grad(pred.sum(),(image,cot),retain_graph=False)
        lo,hi = bounds(size)
        outside = torch.ones((64,64),dtype=torch.bool,device=device)
        outside[lo:hi,lo:hi] = False
        assert (gimage[...,outside] == 0).all() and (gcot[...,outside] == 0).all()
        invalid = torch.arange(24,device=device)[None] > (8+lead[:,None])
        assert (gcot[invalid] == 0).all()
        changed_i = image.detach().clone(); changed_c = cot.detach().clone()
        changed_i[...,outside] = 123.; changed_c[...,outside] = -234.
        changed_c[invalid] = 345.
        with torch.no_grad():
            restored = model(changed_i,changed_c,st,lead,size,True)
        assert torch.equal(pred.detach(),restored)
        checks.append(dict(size=size,outside_input_gradient_max=0,post_target_cot_gradient_max=0,
                           perturbation_prediction_max=0))
    return dict(state='PASS_INPUT_SUPPORT_AND_CAUSAL_PREFIX',checks=checks,
                parameters=sum(p.numel() for p in model.parameters()))


class Bank:
    def __init__(self, bank, base, split, contract, device):
        folder = bank/split
        self.ids = np.load(folder/'head_row_ids.npy')
        seq = np.load(folder/'sequence_indices.npy')
        assert seq.tolist() == contract['selected_sequences'][split]
        lookup = {int(s):i for i,s in enumerate(seq)}
        labels = {k:np.load(base/(k+'.npy'),mmap_mode='r')[self.ids] for k in
                  ('sequence','station','lead','ghi','clear','split')}
        assert (labels['split'] == (0 if split=='train' else 1)).all()
        assert len(np.unique(self.ids)) == len(self.ids)
        self.idx = torch.tensor([lookup[int(s)] for s in labels['sequence']],device=device)
        self.st = torch.tensor(labels['station'].astype('int64'),device=device)
        self.ld = torch.tensor(labels['lead'].astype('int64'),device=device)
        self.ghi = torch.tensor(labels['ghi'],dtype=torch.float64,device=device)
        self.clear = torch.tensor(labels['clear'],dtype=torch.float64,device=device)
        assert torch.isfinite(self.ghi).all() and (self.clear>0).all()
        assert ((self.st>=0)&(self.st<2)&(self.ld>=0)&(self.ld<16)).all()
        self.days = np.array([contract['initialization_BJT_days'][split][str(int(s))] for s in labels['sequence']])
        self.sources = {}
        for k in ('image_local','cot_local_history','cot_local_forecast'):
            path = folder/(k+'.npy')
            a = np.load(path)
            assert np.isfinite(a).all(), k
            self.sources[k] = dict(path=str(path),sha256=sha(path),shape=list(a.shape))
            setattr(self,k,torch.tensor(a,dtype=torch.float16,device=device))
        assert self.image_local.shape == (len(seq),2,16,16,64,64)
        assert self.cot_local_history.shape == (len(seq),2,8,1,64,64)
        assert self.cot_local_forecast.shape == (len(seq),2,16,1,64,64)
        self.n = len(self.ids)

    def batch(self, ids, cot_mean, cot_std):
        ix,st,ld = self.idx[ids],self.st[ids],self.ld[ids]
        image = self.image_local[ix,st,ld].float()
        cot = torch.cat((self.cot_local_history[ix,st,:,0],self.cot_local_forecast[ix,st,:,0]),1).float()
        cot = (cot-cot_mean)/cot_std
        return image,cot,st,ld


@torch.no_grad()
def evaluate(model, data, size, use_cot, batch, cm, cs):
    model.eval(); preds=[]
    for lo in range(0,data.n,batch):
        ids = torch.arange(lo,min(lo+batch,data.n),device=data.ghi.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            kt = model(*data.batch(ids,cm,cs),size,use_cot)
        preds.append((kt.double()*data.clear[ids]).cpu().numpy())
    arr = dict(row_ids=data.ids,pred_ghi=np.concatenate(preds),true_ghi=data.ghi.cpu().numpy(),
               station=data.st.cpu().numpy(),lead_minutes=(data.ld.cpu().numpy()+1)*15,init_day=data.days)
    assert np.isfinite(arr['pred_ghi']).all()
    def metric(mask):
        e = arr['pred_ghi'][mask]-arr['true_ghi'][mask]
        return dict(n=len(e),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(e.mean()))
    stations={str(st):metric(arr['station']==st) for st in (0,1)}
    metrics=dict(station_equal_rmse=sum(m['rmse'] for m in stations.values())/2,stations=stations,
                 leads={str(ld):metric(arr['lead_minutes']==ld) for ld in range(15,241,15)})
    return metrics,arr


def update(model,opt,data,ids,size,ac,cm,cs):
    model.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        p = model(*data.batch(ids,cm,cs),size,ac).float()
    # Equal station weighting, fixed physical GHI scale; observations are targets only.
    err = ((p.double()*data.clear[ids]-data.ghi[ids])/1000).square()
    loss = .5*sum(err[data.st[ids]==st].mean() for st in (0,1))
    assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    opt.step(); return float(loss.detach())


def balanced_order(data, seed_value, steps, batch):
    rng=np.random.default_rng(seed_value)
    groups=[np.flatnonzero(data.st.cpu().numpy()==st) for st in (0,1)]
    streams=[]
    for g in groups:
        stream=np.concatenate([rng.permutation(g) for _ in range(math.ceil(steps*(batch//2)/len(g)))])
        streams.append(stream[:steps*(batch//2)].reshape(steps,batch//2))
    return np.concatenate(streams,1)


def bootstrap(a,b):
    for k in ('row_ids','true_ghi','station','lead_minutes','init_day'): assert np.array_equal(a[k],b[k]),k
    days=np.unique(a['init_day']); mass=np.zeros((len(days),2,3))
    for di,day in enumerate(days):
        for st in (0,1):
            m=(a['init_day']==day)&(a['station']==st)
            mass[di,st]=[m.sum(),np.square(a['pred_ghi'][m]-a['true_ghi'][m]).sum(),
                        np.square(b['pred_ghi'][m]-b['true_ghi'][m]).sum()]
    rng=np.random.default_rng(20261004)
    x=mass[rng.integers(0,len(days),(2000,len(days)))].sum(1)
    x=x[(x[:,:,0]>0).all(1)]
    diff=(np.sqrt(x[:,:,2]/x[:,:,0])-np.sqrt(x[:,:,1]/x[:,:,0])).mean(1)
    return dict(unit='BJT initialization day',days=len(days),repetitions=2000,
                delta_rmse_ci95=np.quantile(diff,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser()
    for k in ('bank','base','output'): p.add_argument('--'+k,type=Path)
    p.add_argument('--witness-only',action='store_true')
    p.add_argument('--updates',type=int,default=1200)
    a=p.parse_args(); torch.set_num_threads(2)
    if a.witness_only:
        print(json.dumps(witness('cpu'))); return
    assert a.bank and a.base and a.output and not a.output.exists()
    a.output.mkdir(parents=True)
    write(a.output/'witness.json',witness('cuda'))
    c=json.loads((a.bank/'complete.json').read_text())
    assert c['state']=='COMPLETE_EXPLORATORY_REGIONAL_GHI_PILOT_BANK' and c['test_used'] is False
    assert sha(a.base/'audit.json')==c['base_pack_audit_sha256']
    assert c['station_rc']==[[163,148],[73,150]] and c['lead_minutes']==list(range(15,241,15))
    write(a.output/'source_contract.json',c)
    tr=Bank(a.bank,a.base,'train',c,'cuda'); va=Bank(a.bank,a.base,'val',c,'cuda')
    # One common train-only normalization from identical full64 COT support.
    cot_arrays=(tr.cot_local_history,tr.cot_local_forecast)
    count=sum(x.numel() for x in cot_arrays)
    cm=sum(float(x.float().sum()) for x in cot_arrays)/count
    cs=math.sqrt(max(sum(float(x.float().square().sum()) for x in cot_arrays)/count-cm*cm,1e-12))
    write(a.output/'data_audit.json',dict(train_rows=tr.n,val_rows=va.n,train_sequences=len(tr.image_local),
          val_sequences=len(va.image_local),sources={'train':tr.sources,'val':va.sources},cot_mean=cm,cot_std=cs,
          source_contract_sha256=sha(a.bank/'complete.json'),script_sha256=sha(Path(__file__)),
          key_hashes={str(path):sha(path) for path in
                     [a.bank/s/(k+'.npy') for s in ('train','val') for k in ('head_row_ids','sequence_indices')]+
                     [a.base/(k+'.npy') for k in ('sequence','station','lead','ghi','clear','split')]},
          cot_norm='common train full64 log1p values',test_used=False,oracle_read=False))
    trials=[]
    total=torch.cuda.get_device_properties(0).total_memory
    for batch in (64,128,256,512,1024):
        model=opt=None
        try:
            seed(42);model=Head().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            order=balanced_order(tr,42,8,batch)
            torch.cuda.reset_peak_memory_stats()
            for step in range(2):update(model,opt,tr,torch.tensor(order[step],device='cuda'),64,True,cm,cs)
            torch.cuda.synchronize();start=time.monotonic()
            for step in range(2,8):update(model,opt,tr,torch.tensor(order[step],device='cuda'),64,True,cm,cs)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            trial=dict(batch=batch,peak_gib=torch.cuda.max_memory_allocated()/2**30,samples_per_second=6*batch/elapsed)
            print('PROFILE',json.dumps(trial),flush=True)
            if trial['peak_gib'] < .85*total/2**30: trials.append(trial)
            else:break
        except torch.cuda.OutOfMemoryError:break
        finally:
            del model,opt;torch.cuda.empty_cache()
    assert trials
    fastest=max(t['samples_per_second'] for t in trials)
    batch=min(t['batch'] for t in trials if t['samples_per_second']>=.95*fastest)
    write(a.output/'profile.json',dict(trials=trials,batch=batch,rule='smallest within5pct fastest,15pct VRAM reserve'))
    records=[];outputs={};init_hashes={};started=time.monotonic()
    for sv in SEEDS:
        order=balanced_order(tr,sv,a.updates,batch)
        for size in WINDOWS:
            for ac in (False,True):
                seed(sv); model=Head().cuda()
                initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
                assert init_hashes.setdefault(sv,initial)==initial
                run=a.output/f"s{size:02d}_{'AC' if ac else 'A'}_seed{sv}";run.mkdir()
                opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
                best=float('inf');curve=[]
                for step in range(1,a.updates+1):
                    lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(step-1)/a.updates))
                    for g in opt.param_groups:g['lr']=lr
                    loss=update(model,opt,tr,torch.tensor(order[step-1],device='cuda'),size,ac,cm,cs)
                    if step%100==0 or step==a.updates:
                        metrics,arr=evaluate(model,va,size,ac,batch,cm,cs)
                        if metrics['station_equal_rmse']<best:
                            best=metrics['station_equal_rmse']
                            torch.save(dict(model=model.state_dict(),step=step,size=size,ac=ac,seed=sv,init_sha256=initial),run/'best.pt')
                            np.savez_compressed(run/'validation_predictions.npz',**arr)
                        curve.append(dict(step=step,loss=loss,lr=lr,**metrics))
                        status=dict(state='TRAINING',run=run.name,step=step,total_runs=36,completed_runs=len(records),best_rmse=best)
                        write(a.output/'status.json',status);write(run/'curve.json',curve)
                        print('PROGRESS',json.dumps(status),flush=True)
                torch.save(dict(model=model.state_dict(),step=a.updates),run/'last.pt')
                final=metrics
                state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(state['model'],strict=True)
                metrics,arr=evaluate(model,va,size,ac,batch,cm,cs)
                with np.load(run/'validation_predictions.npz') as saved:
                    assert all(np.array_equal(saved[k],arr[k]) for k in arr), 'reload mismatch'
                record=dict(window=size,arm='AC' if ac else 'A',seed=sv,best_step=state['step'],metrics=metrics,
                            final_metrics=final,parameters=sum(v.numel() for v in model.parameters()),
                            init_sha256=initial,checkpoint_sha256=sha(run/'best.pt'),prediction_sha256=sha(run/'validation_predictions.npz'))
                records.append(record);outputs[(size,ac,sv)]=arr;write(run/'complete.json',record)
                del model,opt;torch.cuda.empty_cache()
    rows=[]
    for size in WINDOWS:
        row=dict(window=size)
        for ac in (False,True):
            selected=[r['metrics'] for r in records if r['window']==size and (r['arm']=='AC')==ac]
            vals=[m['station_equal_rmse'] for m in selected]
            def aggregate(scope):
                return {key:{metric:dict(mean=float(np.mean([m[scope][key][metric] for m in selected])),
                                       seed_std=float(np.std([m[scope][key][metric] for m in selected],ddof=1)))
                             for metric in ('rmse','mae','bias')} for key in selected[0][scope]}
            row['AC' if ac else 'A']=dict(mean_rmse=float(np.mean(vals)),seed_std=float(np.std(vals,ddof=1)),
                                         seed_values=vals,stations=aggregate('stations'),leads=aggregate('leads'))
        row['AC_minus_A']=row['AC']['mean_rmse']-row['A']['mean_rmse']
        # Average squared errors per row across seeds, retain original targets.
        def seeded_proxy(w,ac):
            arr=dict(outputs[(w,ac,42)])
            e=np.stack([outputs[(w,ac,s)]['pred_ghi']-arr['true_ghi'] for s in SEEDS])
            arr['pred_ghi']=arr['true_ghi']+np.sqrt(np.mean(e*e,0))
            return arr
        row['pooled_seed_mse_AC_minus_A_day_bootstrap']=bootstrap(seeded_proxy(size,False),seeded_proxy(size,True))
        row['pooled_seed_mse_AC_minus_64_day_bootstrap']=bootstrap(seeded_proxy(64,True),seeded_proxy(size,True))
        rows.append(row)
    report=dict(state='COMPLETE_FORECAST_GHI_WINDOW_PILOT',test_used=False,oracle_read=False,
                scope='old Hunan split; frozen forecast AGRI and historical/forecast regional R; validation screening only',
                windows=list(WINDOWS),seeds=list(SEEDS),batch=batch,updates=a.updates,runs=records,rows=rows,
                normalization=dict(cot_mean=cm,cot_std=cs),elapsed_seconds=time.monotonic()-started,
                uncertainty='date bootstrap on per-row seed-averaged squared error; conditional on validation-selected checkpoints')
    write(a.output/'summary.json',report);write(a.output/'status.json',dict(state=report['state'],completed_runs=36))
    print(report['state'],flush=True)


if __name__=='__main__':main()
