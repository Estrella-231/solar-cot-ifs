#!/usr/bin/env python3
"""Four-arm loss-only experiment; source trainer SHA256 a2921f8642b95bb8f9313197529e7bbef0b314fb55afccb4037ed280e668eda8."""
import argparse,csv,hashlib,json,time,random
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
        h=F.silu(self.temporal(x)).squeeze(2);h=F.silu(self.spatial(h))
        return F.softplus(self.fc(h.flatten(1)).squeeze(1))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()

def write_json(p,x):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--r-bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=1792);p.add_argument('--device',default='cuda:0');p.add_argument('--profile-only',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    mp=json.loads((a.pack/'pack_manifest.json').read_text());bm=json.loads((a.r_bank/'bank_manifest.json').read_text())
    if mp['split_id']!='hunan_monthly_block_2d_buffer1d_seed42_v1' or mp['test_payload_opened'] is not False:raise ValueError('GHI pack gate failed')
    if bm.get('state')!='COMPLETE_PAIRED_R_COT_HISTORY_BANK' or bm.get('test_payloads_read')!=0 or bm.get('all_rows_common') is not True:raise ValueError('paired R bank gate failed')
    rows=list(csv.DictReader((a.pack/'rows.csv').open(newline='',encoding='utf-8')));assert len(rows)==51929 and sum(r['split']=='train' for r in rows)==47666 and sum(r['split']=='val' for r in rows)==4263
    if bm.get('rows')!=len(rows) or bm.get('source_GHI_rows_sha256')!=sha(a.pack/'rows.csv'):raise ValueError('paired R bank was built from a different GHI row manifest')
    if {r['split'] for r in rows}!={'train','val'}:raise ValueError('non train/val row')
    if len({(r['station'],r['target_time_utc']) for r in rows})!=len(rows):raise ValueError('duplicate GHI target keys')
    ag=np.load(a.pack/'agri_raw.npy',mmap_mode='r',allow_pickle=False);valid=np.load(a.pack/'agri_valid.npy',mmap_mode='r',allow_pickle=False).astype(bool);geo=np.load(a.pack/'geometry.npy',mmap_mode='r',allow_pickle=False)
    co=np.load(a.r_bank/'cot_old_log1p.npy',mmap_mode='r',allow_pickle=False);cc=np.load(a.r_bank/'cot_corrected_log1p.npy',mmap_mode='r',allow_pickle=False)
    n=len(rows)
    if ag.shape!=(n,3,13,3,3) or valid.shape!=ag.shape or geo.shape!=(n,3,2,3,3) or co.shape!=(n,3,1,3,3) or cc.shape!=co.shape:raise ValueError('three-arm input shapes differ')
    if not np.isfinite(ag).all() or not np.isfinite(geo).all() or not np.isfinite(co).all() or not np.isfinite(cc).all():raise ValueError('nonfinite input')
    tr=np.asarray([r['split']=='train' for r in rows]);va=~tr;stations=np.asarray([r['station'] for r in rows]);
    am=np.asarray(mp['AGRI_norm_mean'],np.float32)[None,None,:,None,None];ast=np.asarray(mp['AGRI_norm_std'],np.float32)[None,None,:,None,None]
    x=np.where(valid,(np.asarray(ag,np.float32)-am)/ast,0.).astype(np.float32)
    g=np.asarray(geo,np.float32).copy()
    # One common train-only scaler over both paired frozen-R outputs.
    pool=np.concatenate([np.asarray(co[tr],np.float64).reshape(-1),np.asarray(cc[tr],np.float64).reshape(-1)])
    cmean=float(pool.mean());cstd=max(float(pool.std()),1e-6)
    cold=((np.asarray(co,np.float32)-cmean)/cstd).astype(np.float32)
    cnew=((np.asarray(cc,np.float32)-cmean)/cstd).astype(np.float32)
    y=np.asarray([float(r['ghi']) for r in rows],np.float32);clear=np.asarray([float(r['clear']) for r in rows],np.float32)
    if np.any(clear<0) or not np.isfinite(y).all() or not np.isfinite(clear).all():raise ValueError('invalid labels or clear sky')
    arms=('A_uniform','A_bright2x','AC_uniform','AC_bright2x')
    cin={'A_uniform':np.zeros_like(cold),'A_bright2x':np.zeros_like(cold),'AC_uniform':cnew,'AC_bright2x':cnew}
    weights=np.ones(n,np.float32)
    high_train=tr & (np.array([float(r['clear']) for r in rows])>=100) & (np.array([float(r['kt']) for r in rows])>=.9)
    weights[high_train]=2
    train_weight_receipt={st:dict(train_n=int((tr&(stations==st)).sum()),bright_n=int((high_train&(stations==st)).sum()),mean_raw_weight=float(weights[tr&(stations==st)].mean())) for st in ('sili','zhujia')}
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    dev=torch.device(a.device);X=torch.from_numpy(x).to(dev);G=torch.from_numpy(g).to(dev);C={k:torch.from_numpy(v).to(dev) for k,v in cin.items()};Y=torch.from_numpy(y).to(dev);CL=torch.from_numpy(clear).to(dev)
    # Witness: identical init functions; both COT channels can affect the output.
    torch.manual_seed(42);base=IntervalHead().to(dev);init={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    testids=torch.arange(min(128,n),device=dev);p0=base(X[testids],G[testids],C[arms[0]][testids]);diffs={k:float((base(X[testids],G[testids],C[k][testids])-p0).abs().max()) for k in arms[1:]}
    if max(diffs.values())!=0.:raise AssertionError(f'COT arms not function-matched at initialization: {diffs}')
    for arm in ('AC_uniform','AC_bright2x'):
        probe=IntervalHead().to(dev);probe.load_state_dict(init);probe.zero_grad(set_to_none=True);pred=probe(X[testids],G[testids],C[arm][testids]);(((pred*CL[testids]-Y[testids])/1000).square().mean()).backward()
        grad=float(probe.temporal.weight.grad[:,15].norm());
        if grad<=0:raise AssertionError(f'COT channel disconnected: {arm}')
    param=sum(q.numel() for q in base.parameters());del base
    write_json(a.output/'setup.json',dict(state='PASS_FOUR_ARM_LOSS_SETUP',rows=n,train=int(tr.sum()),val=int(va.sum()),history_offsets_minutes=[-30,-15,0],arms=list(arms),parameters_each_arm=param,train_weight_receipt=train_weight_receipt,loss_weight='2 if train clear>=100 and observed_kt>=0.9 else 1; station-train mean normalized',batch=a.batch,epochs=30,seeds=[42,43,44],stations=['sili','zhujia'],initial_output_max_abs_difference=diffs,cot_channel_grad_witness='PASS',COT_pool_train_mean=cmean,COT_pool_train_std=cstd,AGRI_normalization_source='fixed data_attempt2 train-only normalization',geometry='common corrected nominal-UTC geometry from fixed pack',frozen_R_models={'scaler_old_source':bm['old_R_checkpoint_sha256'],'corrected':bm['corrected_R_checkpoint_sha256']},bank_manifest_sha256=sha(a.r_bank/'bank_manifest.json'),pack_rows_sha256=sha(a.pack/'rows.csv'),test_payloads_read=0))
    # Forward/backward profile uses the exact training batch and no model-selection data.
    import platform
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(42)
    profile_model=IntervalHead().to(dev);profile_opt=torch.optim.AdamW(profile_model.parameters(),lr=.001,weight_decay=.0001)
    pi=torch.as_tensor(np.flatnonzero(tr)[:a.batch],device=dev)
    for _ in range(3):
        profile_opt.zero_grad(set_to_none=True);pl=((profile_model(X[pi],G[pi],C['AC_uniform'][pi])*CL[pi]-Y[pi])/1000).square().mean();pl.backward();profile_opt.step()
    torch.cuda.synchronize();pstart=time.time()
    for _ in range(20):
        profile_opt.zero_grad(set_to_none=True);pl=((profile_model(X[pi],G[pi],C['AC_uniform'][pi])*CL[pi]-Y[pi])/1000).square().mean();pl.backward();profile_opt.step()
    torch.cuda.synchronize();seconds=time.time()-pstart
    profile=dict(state='PASS_ACTUAL_FORWARD_BACKWARD_OPTIMIZER',device=torch.cuda.get_device_name(dev),torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),batch=a.batch,steps=20,seconds=seconds,samples_per_second=20*len(pi)/seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finite_loss=bool(torch.isfinite(pl)),test_payloads_read=0)
    assert profile['finite_loss'];write_json(a.output/'profile.json',profile);print('PROFILE_COMPLETE '+json.dumps(profile),flush=True)
    del profile_model,profile_opt
    if a.profile_only: return
    allstats=[];started=time.time()
    for seed in (42,43,44):
        rng=np.random.default_rng(seed)
        for st in ('sili','zhujia'):
            trainids=np.where(tr&(stations==st))[0];validids=np.where(va&(stations==st))[0]
            orders=[rng.permutation(trainids) for _ in range(30)]
            torch.manual_seed(seed);base=IntervalHead().to(dev);initial={k:v.detach().cpu().clone() for k,v in base.state_dict().items()};del base
            for arm in arms:
                folder=a.output/f'seed{seed}'/st/arm;folder.mkdir(parents=True)
                torch.manual_seed(seed);model=IntervalHead().to(dev);model.load_state_dict(initial,strict=True)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
                best=float('inf');beststate=None;logs=[];updates=0;start=time.time();ci=C[arm];vi=torch.as_tensor(validids,device=dev)
                normalized_weights=weights/float(weights[trainids].mean()) if arm.endswith('bright2x') else np.ones(n,np.float32)
                W=torch.from_numpy(normalized_weights).to(dev)
                assert torch.isfinite(W).all() and (W>0).all()
                for ep,order in enumerate(orders):
                    model.train();loss_sum=0.
                    for pos in range(0,len(order),a.batch):
                        ii=torch.as_tensor(order[pos:pos+a.batch],device=dev);pred=model(X[ii],G[ii],ci[ii])*CL[ii]
                        loss=(W[ii]*((pred-Y[ii])/1000).square()).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();updates+=1;loss_sum+=float(loss.detach())*len(ii)
                    sched.step();model.eval()
                    with torch.inference_mode():pv=(model(X[vi],G[vi],ci[vi])*CL[vi]).cpu().numpy()
                    rmse=float(np.sqrt(np.mean((pv.astype(np.float64)-y[validids].astype(np.float64))**2)))
                    rec=dict(epoch=ep,val_rmse=rmse,train_mse_scaled=loss_sum/len(trainids),updates=updates);logs.append(rec)
                    if rmse<best:best=rmse;best_ep=ep;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if ep==0 or (ep+1)%10==0:print(json.dumps(dict(seed=seed,station=st,arm=arm,**rec)),flush=True)
                model.load_state_dict(beststate);model.eval()
                with torch.inference_mode():pred=(model(X[vi],G[vi],ci[vi])*CL[vi]).cpu().numpy().astype(np.float64)
                ck=dict(model=beststate,seed=seed,station=st,arm=arm,best_epoch=best_ep,parameters=param,train_n=len(trainids),val_n=len(validids),pack_manifest_sha256=sha(a.pack/'pack_manifest.json'),R_bank_manifest_sha256=sha(a.r_bank/'bank_manifest.json'),COT_norm_mean=cmean,COT_norm_std=cstd,test_payloads_read=0)
                torch.save(ck,folder/'best.pt')
                cpu=IntervalHead();cpu.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model']);cpu.eval()
                with torch.inference_mode():pc=(cpu(X[vi].cpu(),G[vi].cpu(),ci[vi].cpu())*CL[vi].cpu()).numpy().astype(np.float64)
                delta=float(np.max(np.abs(pc-pred)));assert delta<.1,delta
                indices=validids.astype(np.int64);np.savez_compressed(folder/'validation_predictions.npz',prediction=pred,truth=y[validids].astype(np.float64),pack_index=indices,cpu_prediction=pc)
                rec=dict(seed=seed,station=st,arm=arm,n=len(validids),train_n=len(trainids),rmse=float(np.sqrt(np.mean((pred-y[validids])**2))),mae=float(np.mean(np.abs(pred-y[validids]))),mbe=float(np.mean(pred-y[validids])),best_epoch=best_ep,updates=updates,seconds=time.time()-start,cpu_reload_max_abs_GHI=delta,checkpoint_sha256=sha(folder/'best.pt'),prediction_sha256=sha(folder/'validation_predictions.npz'))
                write_json(folder/'metrics.json',rec);(folder/'training.json').write_text(json.dumps(logs,indent=2)+'\n');allstats.append(rec);print('ARM_COMPLETE '+json.dumps(rec),flush=True)
    valids=np.where(va)[0]
    with (a.output/'validation_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows[i] for i in valids)
    np.savez_compressed(a.output/'validation_inputs.npz',pack_index=valids.astype(np.int64),agri_raw=np.asarray(ag[valids]),agri_valid=np.asarray(valid[valids]),geometry=np.asarray(geo[valids]),cot_old_log1p=np.asarray(co[valids]),cot_corrected_log1p=np.asarray(cc[valids]))
    write_json(a.output/'completion.json',dict(state='COMPLETE_24_LOSS_PAIRED_RUNS',runs=allstats,elapsed_seconds=time.time()-started,test_payloads_read=0))
    print('FOUR_ARM_LOSS_PAIRED_GHI_COMPLETE',flush=True)
if __name__=='__main__':main()
