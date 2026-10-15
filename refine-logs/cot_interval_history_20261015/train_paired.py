"""Three-frame local AGRI versus AGRI+continuous-COT, paired monthly validation."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class IntervalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal=nn.Conv3d(16,32,(3,3,3),padding=(0,1,1))
        self.spatial=nn.Conv2d(32,32,3,padding=1)
        self.fc=nn.Sequential(nn.Linear(32*3*3,64),nn.SiLU(),nn.Linear(64,1))
        with torch.no_grad():self.temporal.weight[:,15].zero_()

    def forward(self,ag,geo,cot):
        x=torch.cat((ag,geo,cot),dim=2).permute(0,2,1,3,4)
        h=F.silu(self.temporal(x)).squeeze(2)
        h=F.silu(self.spatial(h))
        return F.softplus(self.fc(h.flatten(1)).squeeze(1))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device=torch.device('cuda');meta=json.loads((a.pack/'pack_manifest.json').read_text())
    assert meta['test_payload_opened'] is False and meta['split_id']=='hunan_monthly_block_2d_buffer1d_seed42_v1'
    rows=list(csv.DictReader((a.pack/'rows.csv').open()));assert {r['split'] for r in rows}<={'train','val'}
    ag=np.load(a.pack/'agri_raw.npy');valid=np.load(a.pack/'agri_valid.npy').astype(bool)
    am=np.asarray(meta['AGRI_norm_mean'])[None,None,:,None,None];ast=np.asarray(meta['AGRI_norm_std'])[None,None,:,None,None]
    x=np.where(valid,(ag-am)/ast,0).astype(np.float32);g=np.load(a.pack/'geometry.npy').astype(np.float32)
    c=((np.load(a.pack/'cot_log1p.npy').astype(np.float32)-meta['COT_norm_mean'])/meta['COT_norm_std'])
    if c.ndim==4:c=c[:,:,None]
    assert x.shape[1:]==(3,13,3,3) and g.shape[1:]==(3,2,3,3) and c.shape[1:]==(3,1,3,3)
    assert np.isfinite(x).all() and np.isfinite(c).all() and np.isfinite(g).all()
    X=torch.from_numpy(x).to(device);G=torch.from_numpy(g).to(device);C=torch.from_numpy(c).to(device);Z=torch.zeros_like(C)
    y=np.asarray([float(r['ghi']) for r in rows],np.float32);cl=np.asarray([float(r['clear']) for r in rows],np.float32)
    Y=torch.from_numpy(y).to(device);CL=torch.from_numpy(cl).to(device)
    # Seeded actual forward/backward: identical initial functions and both historical slots connected.
    torch.manual_seed(42);m=IntervalHead().to(device);ids=torch.arange(128,device=device)
    xx=X[ids].detach().clone().requires_grad_();gg=G[ids];cc=C[ids]
    p=m(xx,gg,cc);p0=m(xx,gg,Z[ids]);initial_difference=float((p-p0).abs().max());assert initial_difference==0
    ((p*CL[ids]-Y[ids])/1000).square().mean().backward()
    cot_grad=float(m.temporal.weight.grad[:,15].norm());history_grads=xx.grad.abs().sum((0,2,3,4)).cpu().tolist()
    assert cot_grad>0 and min(history_grads)>0
    profile=[]
    trmin=min(sum(r['station']==st and r['split']=='train' for r in rows) for st in ('sili','zhujia'))
    for b in (256,512,1024,1536,1792,2000):
        if b>trmin//12:continue
        torch.manual_seed(42);m=IntervalHead().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        ii=torch.arange(b,device=device)
        for _ in range(3):
            loss=((m(X[ii],G[ii],C[ii])*CL[ii]-Y[ii])/1000).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        for _ in range(12):
            loss=((m(X[ii],G[ii],C[ii])*CL[ii]-Y[ii])/1000).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        torch.cuda.synchronize();secs=time.perf_counter()-t
        profile.append(dict(batch=b,examples_per_second=b*12/secs,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
    bestspeed=max(r['examples_per_second'] for r in profile);batch=max(r['batch'] for r in profile if r['examples_per_second']>=.97*bestspeed)
    setup=dict(state='PROFILE_AND_ACTUAL_KERNEL_WITNESS_PASS',profile=profile,selected_batch=batch,initial_function_difference=initial_difference,
        COT_input_weight_gradient_norm=cot_grad,AGRI_gradient_by_time=history_grads,parameters=sum(p.numel() for p in m.parameters()),GPU=torch.cuda.get_device_name(),
        torch_version=torch.__version__,geometry_contract=meta['geometry'],test_payload_opened=False,
        source_hashes={n:sha(a.pack/n) for n in ('pack_manifest.json','rows.csv','agri_raw.npy','agri_valid.npy','cot_log1p.npy','geometry.npy')},
        code_sha256=sha(__file__))
    (a.output/'profile_and_setup.json').write_text(json.dumps(setup,indent=2));print(json.dumps(setup),flush=True)
    allstats=[];training_start=time.time()
    for seed in (42,43,44):
        for station in ('sili','zhujia'):
            tr=np.asarray([i for i,r in enumerate(rows) if r['station']==station and r['split']=='train']);va=np.asarray([i for i,r in enumerate(rows) if r['station']==station and r['split']=='val'])
            vi=torch.as_tensor(va,device=device);rng=np.random.default_rng(seed);orders=[rng.permutation(tr) for _ in range(30)]
            torch.manual_seed(seed);base=IntervalHead().state_dict();initial={k:v.clone() for k,v in base.items()}
            for arm in ('A_history','AC_history'):
                folder=a.output/f'seed{seed}'/station/arm;folder.mkdir(parents=True)
                torch.manual_seed(seed);model=IntervalHead().to(device);model.load_state_dict(initial)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
                cot=C if arm=='AC_history' else Z;best=float('inf');logs=[];start=time.time();updates=0
                for ep,order in enumerate(orders):
                    model.train();loss_sum=0
                    for k in range(0,len(order),batch):
                        ii=torch.as_tensor(order[k:k+batch],device=device);pred=model(X[ii],G[ii],cot[ii])*CL[ii]
                        loss=((pred-Y[ii])/1000).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();updates+=1;loss_sum+=float(loss.detach())*len(ii)
                    sched.step();model.eval()
                    with torch.inference_mode():pv=(model(X[vi],G[vi],cot[vi])*CL[vi]).cpu().numpy()
                    rmse=float(np.sqrt(np.mean((pv.astype(float)-y[va].astype(float))**2)))
                    rec=dict(epoch=ep,val_rmse=rmse,train_mse_scaled=loss_sum/len(tr),updates=updates);logs.append(rec)
                    if rmse<best:
                        best=rmse;best_ep=ep;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if ep==0 or (ep+1)%10==0:print(json.dumps(dict(seed=seed,station=station,arm=arm,**rec)),flush=True)
                model.load_state_dict(state);model.eval()
                with torch.inference_mode():pv=(model(X[vi],G[vi],cot[vi])*CL[vi]).cpu().numpy().astype(float)
                torch.save(dict(model=state,meta=meta,arm=arm,seed=seed,best_epoch=best_ep),folder/'best.pt')
                cpu=IntervalHead();cpu.load_state_dict(torch.load(folder/'best.pt',map_location='cpu')['model']);cpu.eval()
                with torch.inference_mode():pc=(cpu(torch.from_numpy(x[va]),torch.from_numpy(g[va]),torch.from_numpy(c[va]) if arm=='AC_history' else torch.zeros_like(torch.from_numpy(c[va])))*torch.from_numpy(cl[va])).numpy().astype(float)
                err=float(np.abs(pc-pv).max());assert err<.1,err
                np.savez_compressed(folder/'validation_predictions.npz',prediction=pv,truth=y[va].astype(float),pack_index=va,cpu_prediction=pc)
                stat=dict(seed=seed,station=station,arm=arm,rmse=float(np.sqrt(np.mean((pv-y[va])**2))),mae=float(np.mean(np.abs(pv-y[va]))),n=len(va),train_n=len(tr),best_epoch=best_ep,updates=updates,seconds=time.time()-start,cpu_reload_max_abs_GHI=err,checkpoint_sha256=sha(folder/'best.pt'))
                (folder/'metrics.json').write_text(json.dumps(stat,indent=2));(folder/'training.json').write_text(json.dumps(logs));allstats.append(stat);print('ARM_COMPLETE '+json.dumps(stat),flush=True)
    val=np.asarray([i for i,r in enumerate(rows) if r['split']=='val']);np.savez_compressed(a.output/'validation_inputs.npz',agri_raw=ag[val],cot_log1p=np.load(a.pack/'cot_log1p.npy')[val],geometry=g[val],pack_index=val)
    with (a.output/'validation_rows.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows[i] for i in val)
    (a.output/'completion.json').write_text(json.dumps(dict(state='COMPLETE_TWELVE_PAIRED_RUNS',runs=allstats,elapsed_seconds=time.time()-training_start,test_payload_opened=False),indent=2));print('INTERVAL_HISTORY_PAIRED_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
