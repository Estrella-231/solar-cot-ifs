#!/usr/bin/env python3
"""Train one T18 AL head under the existing matched GHI-head contract."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from head_candidate_v0 import LocalGHIHead
from head_r_al_candidate import LocalGHIHeadAL,parameter_count

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()
def read_rows(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    for k in ('pack','feature-bank','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--station',choices=('sili','zhujia'),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=2000)
    p.add_argument('--profile-only',action='store_true',help='profile real AL updates only; do not create a training run')
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('formal AL training requires CUDA')
    if not a.profile_only and a.output.exists():raise FileExistsError(a.output)
    pm=json.loads((a.pack/'pack_manifest.json').read_text());fm=json.loads((a.feature_bank/'feature_manifest.json').read_text())
    if pm.get('state')!='COMPLETE' or fm.get('state')!='COMPLETE' or pm.get('test_payload_opened') is not False or fm.get('test_payload_opened') is not False:
        raise ValueError('GHI/feature manifest gate failed')
    rows=read_rows(a.pack/'rows.csv');feature_rows=read_rows(a.feature_bank/'frame_keys.csv')
    frame_idx={r['target_time_utc']:int(r['index']) for r in feature_rows}
    station_idx=('sili','zhujia').index(a.station)
    fi=np.asarray([frame_idx.get(r['target_time_utc'],-1) if r['station']==a.station else 0 for r in rows],np.int64)
    keep=np.asarray([r['station']==a.station for r in rows]);train=np.asarray([r['split']=='train' for r in rows])&keep;val=np.asarray([r['split']=='val' for r in rows])&keep
    if np.any(fi[keep]<0) or not train.any() or not val.any():raise ValueError('missing station features or split')
    dev=torch.device('cuda');torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    ag=np.load(a.pack/'agri_raw.npy',mmap_mode='r',allow_pickle=False);valid=np.load(a.pack/'agri_valid.npy',mmap_mode='r',allow_pickle=False)
    geo=np.load(a.pack/'geometry.npy',mmap_mode='r',allow_pickle=False);raw_f=np.load(a.feature_bank/'dec1_station_3x3.npy',mmap_mode='r',allow_pickle=False)
    fmnorm=np.load(a.feature_bank/'train_feature_norm.npz',allow_pickle=False);fmean=np.asarray(fmnorm['mean'],np.float32)[None,:,None,None];fstd=np.asarray(fmnorm['std'],np.float32)[None,:,None,None]
    # Build keyed feature tensor in GHI-row order; only current station rows are used.
    f=np.zeros((len(rows),32,3,3),np.float32)
    for i in np.flatnonzero(keep):f[i]=raw_f[fi[i],station_idx]
    f=(f-fmean)/fstd
    am=np.asarray(pm['AGRI_norm_mean'],np.float32)[None,:,None,None];ast=np.asarray(pm['AGRI_norm_std'],np.float32)[None,:,None,None]
    X=np.where(np.asarray(valid,dtype=bool),(np.asarray(ag,dtype=np.float32)-am)/ast,0).astype(np.float32)
    G=np.array(geo,dtype=np.float32,copy=True)
    gh=np.asarray([float(r['ghi']) for r in rows],np.float32);cl=np.asarray([float(r['clear']) for r in rows],np.float32)
    X=torch.from_numpy(X).to(dev);G=torch.from_numpy(G).to(dev);F=torch.from_numpy(f).to(dev)
    GH=torch.from_numpy(gh).to(dev);CL=torch.from_numpy(cl).to(dev)
    tr=np.flatnonzero(train);va=np.flatnonzero(val)
    # Exact paired initialization recipe used by A/AC/AP: copy the A trunk and
    # common readout columns, leave the AP-capacity auxiliary branch initialized
    # from the same seed, then train only this new arm.
    torch.manual_seed(a.seed);base=LocalGHIHead('A',2).to(dev);base_state={k:v.detach().clone() for k,v in base.state_dict().items()}
    torch.manual_seed(a.seed);model=LocalGHIHeadAL(2).to(dev)
    with torch.no_grad():
        model.agri_encoder.load_state_dict(base.agri_encoder.state_dict())
        model.readout[0].weight.zero_();model.readout[0].weight[:,:base.readout[0].in_features].copy_(base.readout[0].weight)
        model.readout[0].bias.copy_(base.readout[0].bias);model.readout[2:].load_state_dict(base.readout[2:].state_dict())
    if sum(p.numel() for p in model.parameters())!=parameter_count():raise RuntimeError('AL parameter count drift')
    if a.profile_only:
        import gc
        candidates=(512,1000,1600,2000)
        records=[];nstation=int(train.sum());total_mem=torch.cuda.get_device_properties(dev).total_memory
        for bs in candidates:
            if int(np.ceil(nstation/bs))<12:
                records.append({'batch_size':bs,'status':'below_12_updates_per_epoch'});continue
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(dev)
            optp=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            order=tr[:12*bs] if len(tr)>=12*bs else np.resize(tr,12*bs)
            t0=time.perf_counter();ok=True
            try:
                model.train()
                for j in range(12):
                    ix=torch.as_tensor(order[j*bs:(j+1)*bs],device=dev)
                    optp.zero_grad(set_to_none=True);pred=model(X[ix],G[ix],F[ix])
                    loss=(((pred*CL[ix]-GH[ix])/1000.)**2).mean();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optp.step()
                torch.cuda.synchronize(dev)
            except torch.cuda.OutOfMemoryError:
                ok=False;optp.zero_grad(set_to_none=True);del ix,pred,loss;torch.cuda.empty_cache()
            elapsed=time.perf_counter()-t0
            peak=int(torch.cuda.max_memory_allocated(dev)) if ok else None
            records.append({'batch_size':bs,'status':'ok' if ok else 'oom','updates':12 if ok else 0,
                            'seconds_per_update':elapsed/12 if ok else None,
                            'samples_per_second':12*bs/elapsed if ok else None,
                            'peak_allocated_bytes':peak,'fraction_total_memory':peak/total_mem if peak else None})
            del optp;gc.collect()
        good=[r for r in records if r['status']=='ok']
        if not good:raise RuntimeError('no candidate batch completed 12 real AL updates')
        best=max(r['samples_per_second'] for r in good)
        eligible=[r for r in good if r['samples_per_second']>=.97*best and r['fraction_total_memory']<=.90]
        if not eligible:eligible=[r for r in good if r['fraction_total_memory']<=.90]
        if not eligible:raise RuntimeError('no stable-memory AL profile candidate')
        selected=max(eligible,key=lambda r:r['batch_size'])
        result={'state':'PASS_SINGLE_CARD_AL_PROFILE','station':a.station,'seed':a.seed,'training_rows':nstation,
                'updates_per_epoch_at_selected_batch':int(np.ceil(nstation/selected['batch_size'])),
                'selected_batch_size':selected['batch_size'],'candidates':records,'test_payload_opened':False}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True);return
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5)
    rng=np.random.default_rng(a.seed);orders=[rng.permutation(tr) for _ in range(a.epochs)]
    a.output.mkdir(parents=True)
    meta={'arm':'AL','station':a.station,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,
          'split_id':pm['split_id'],'sample_key':'(station,target_time_utc)','training_rows':len(tr),'validation_rows':len(va),
          'GHI_pack_manifest_sha256':sha(a.pack/'pack_manifest.json'),'feature_manifest_sha256':sha(a.feature_bank/'feature_manifest.json'),
          'R_checkpoint_sha256':fm['R_checkpoint_sha256'],'R_source_sha256':fm['R_source_sha256'],
          'head_source_sha256':sha(Path(__file__).with_name('head_r_al_candidate.py')),'trainer_sha256':sha(Path(__file__)),
          'base_head_source_sha256':sha(Path(__file__).with_name('head_candidate_v0.py')),
          'initialization':'A shared trunk/readout copied identically; AL 32-to-4 auxiliary branch seeded as AP-capacity branch',
          'optimizer':'AdamW(lr=1e-3,weight_decay=1e-4)','scheduler':'CosineAnnealingLR(T_max=30,eta_min=1e-5)',
          'loss':'physical GHI MSE divided by 1000','feature_normalization':'train-only across keyed GHI training rows from both stations',
          'parameter_count':parameter_count(),'test_payload_opened':False}
    (a.output/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    best=float('inf');best_state=None;log=[];start=time.time()
    for ep,order in enumerate(orders):
        model.train();loss_sum=0.
        for st in range(0,len(order),a.batch_size):
            ix=torch.as_tensor(order[st:st+a.batch_size],device=dev)
            pred=model(X[ix],G[ix],F[ix]);err=(pred*CL[ix]-GH[ix])/1000.;loss=err.square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        sched.step();model.eval();pv=[]
        with torch.inference_mode():
            for st in range(0,len(va),a.batch_size):
                ix=torch.as_tensor(va[st:st+a.batch_size],device=dev);pv.append((model(X[ix],G[ix],F[ix])*CL[ix]).cpu().numpy())
        pred=np.concatenate(pv);truth=gh[va];rmse=float(np.sqrt(np.mean((pred-truth)**2)));mae=float(np.mean(np.abs(pred-truth)))
        rec={'arm':'AL','epoch':ep,'train_mse_scaled':loss_sum/len(tr),'val_ghi_rmse_wm2':rmse,'val_ghi_mae_wm2':mae,'lr':opt.param_groups[0]['lr']};log.append(rec)
        if rmse<best:best=rmse;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep in (0,a.epochs-1) or (ep+1)%5==0:print(json.dumps(rec),flush=True)
    model.load_state_dict(best_state);model.eval();preds=[]
    with torch.inference_mode():
        for st in range(0,len(va),a.batch_size):
            ix=torch.as_tensor(va[st:st+a.batch_size],device=dev);preds.append((model(X[ix],G[ix],F[ix])*CL[ix]).cpu().numpy())
    pred=np.concatenate(preds).astype(np.float64);truth=gh[va].astype(np.float64)
    np.save(a.output/'AL_val_prediction_ghi_wm2.npy',pred);torch.save({'model':best_state,'arm':'AL','seed':a.seed,'best_val_rmse_wm2':best},a.output/'AL_best.pt')
    (a.output/'AL_validation.json').write_text(json.dumps({'arm':'AL','station':a.station,'seed':a.seed,'n':len(va),'best_val_rmse_wm2':float(np.sqrt(np.mean((pred-truth)**2))),'mae_wm2':float(np.mean(np.abs(pred-truth))),'best_epoch':min(r['epoch'] for r in log if r['val_ghi_rmse_wm2']==best),'checkpoint_sha256':sha(a.output/'AL_best.pt'),'test_payload_opened':False},indent=2)+'\n')
    with (a.output/'metrics.jsonl').open('w') as f:
        for r in log:f.write(json.dumps(r)+'\n')
    print('AL_TRAIN_COMPLETE',json.dumps({'station':a.station,'seed':a.seed,'best_val_rmse_wm2':best,'elapsed_seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
