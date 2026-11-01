#!/usr/bin/env python3
"""Train the registered seed-42 cloud gate on the frozen monthly-R train/val pack."""
import argparse, csv, hashlib, json, math, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class CloudGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 1, 1))

    def forward(self, x): return self.net(x)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def metrics(prob, y, valid):
    p=prob[valid].astype(np.float64); yy=y[valid].astype(bool)
    pred=p>=.5
    tp=int((pred&yy).sum()); tn=int((~pred&~yy).sum()); fp=int((pred&~yy).sum()); fn=int((~pred&yy).sum())
    pod=tp/(tp+fn) if tp+fn else float('nan'); far=fp/(tp+fp) if tp+fp else float('nan')
    csi=tp/(tp+fp+fn) if tp+fp+fn else float('nan')
    tpr=pod; tnr=tn/(tn+fp) if tn+fp else float('nan')
    return dict(n=int(valid.sum()),tp=tp,tn=tn,fp=fp,fn=fn,POD=pod,FAR=far,CSI=csi,KSS=tpr+tnr-1,
                Brier=float(np.mean((p-yy.astype(float))**2)))


def main():
    p=argparse.ArgumentParser()
    for k in ('pack','clp','normalization','r-validation-prediction','output'): p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=512)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True; seed_all(a.seed)
    device=torch.device(a.device)
    if device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('registered run requires one CUDA device')
    torch.cuda.set_device(device)
    rows=list(csv.DictReader((a.pack/'rows.csv').open(newline='',encoding='utf-8')))
    rows=[r for r in rows if r['state']=='SUPERVISED']
    with np.load(a.clp,allow_pickle=False) as z:
        if not np.array_equal(z['sample_indices'],np.arange(len(rows))): raise ValueError('CLP sidecar row keys/index order mismatch')
        if not np.array_equal(z['station'],np.asarray([r['station'] for r in rows])) or not np.array_equal(z['target_time_utc'],np.asarray([r['target_time_utc'] for r in rows])):
            raise ValueError('CLP sidecar station/time keys mismatch')
        clp=z['clp'][:,0].astype(np.uint8)
    xraw=np.load(a.pack/'x_raw.npy',mmap_mode='r',allow_pickle=False)
    msk=np.load(a.pack/'mask.npy',mmap_mode='r',allow_pickle=False)[:,0].astype(bool)
    cot=np.load(a.pack/'cot.npy',mmap_mode='r',allow_pickle=False)[:,0].astype(np.float32)
    norm=json.loads(a.normalization.read_text())
    mean=np.asarray(norm['mean'],np.float32)[:,None,None];std=np.asarray(norm['std'],np.float32)[:,None,None]
    if xraw.shape!=(len(rows),16,16,16) or msk.shape!=clp.shape or cot.shape!=clp.shape or mean.shape!=(16,1,1) or (std<=0).any():
        raise ValueError(f'input shape/normalization mismatch: {xraw.shape} {msk.shape} {clp.shape} {mean.shape}')
    valid=msk & np.isin(clp,(0,1,2))
    if not np.array_equal(valid,msk): raise ValueError('R mask has pixels without valid CLP')
    y=(clp>=1).astype(np.float32)
    train=np.asarray([i for i,r in enumerate(rows) if r['split']=='train'],np.int64)
    val=np.asarray([i for i,r in enumerate(rows) if r['split']=='val'],np.int64)
    if len(train)!=15414 or len(val)!=1398: raise ValueError(f'unexpected frozen support: {len(train)}/{len(val)}')
    with np.load(a.r_validation_prediction,allow_pickle=False) as z:
        if not np.array_equal(z['sample_indices'],val): raise ValueError('frozen R validation prediction indices differ from pack')
        rlog=z['pred_log1p_cot'][:,0].astype(np.float32)
    if rlog.shape!=cot[val].shape: raise ValueError('frozen R validation COT shape mismatch')
    x=np.asarray(xraw,np.float32).copy();x=(x-mean[None])/std[None];x[~np.isfinite(x)]=0
    xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(y[:,None]).to(device);mt=torch.from_numpy(valid[:,None]).to(device)
    tr=torch.as_tensor(train,device=device);va=torch.as_tensor(val,device=device)
    # Single-card throughput/memory witness at the registered effective batch.
    prof=CloudGate().to(device);popt=torch.optim.AdamW(prof.parameters(),lr=1e-3,weight_decay=1e-4)
    pix=tr[:min(a.batch_size,len(train))]
    for _ in range(3):
        pl=nn.functional.binary_cross_entropy_with_logits(prof(xt[pix]),yt[pix],reduction='none')
        pvld=mt[pix].float();pl=((pl*pvld).sum((1,2,3))/pvld.sum((1,2,3)).clamp_min(1)).mean()
        popt.zero_grad(set_to_none=True);pl.backward();popt.step()
    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);t0=time.monotonic()
    for _ in range(20):
        pl=nn.functional.binary_cross_entropy_with_logits(prof(xt[pix]),yt[pix],reduction='none')
        pvld=mt[pix].float();pl=((pl*pvld).sum((1,2,3))/pvld.sum((1,2,3)).clamp_min(1)).mean()
        popt.zero_grad(set_to_none=True);pl.backward();popt.step()
    torch.cuda.synchronize(device);profile=dict(device_name=torch.cuda.get_device_name(device),
        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,batch_size=len(pix),
        profile_steps=20,seconds=time.monotonic()-t0,steps_per_second=20/(time.monotonic()-t0),
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),reserved_memory_bytes=torch.cuda.max_memory_reserved(device))
    (a.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    del prof,popt,pl,pvld
    torch.cuda.empty_cache();seed_all(a.seed)
    model=CloudGate().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5)
    updates=a.epochs*math.ceil(len(train)/a.batch_size)
    metadata=dict(contract='monthly_R_station16_cloud_gate_seed42_v1',seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,
                  expected_updates=updates,train_samples=len(train),val_samples=len(val),device=str(device),
                  model='Conv2d 16->16->32->16->1; 3x3 SiLU x3, 1x1 logit; no pooling',
                  class_target='CLP=0 clear; CLP=1/2 cloud',input_norm_sha256=sha(a.normalization),
                  R_pack_status_sha256=sha(a.pack/'pack_status.json'),clp_sidecar_sha256=sha(a.clp),
                  R_validation_prediction_sha256=sha(a.r_validation_prediction),test_payloads_read=0)
    (a.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    history=[];best=-float('inf');best_epoch=None
    bce=nn.BCEWithLogitsLoss(reduction='none')
    start=time.monotonic();torch.cuda.reset_peak_memory_stats(device);updates_done=0
    for ep in range(1,a.epochs+1):
        model.train();perm=tr[torch.randperm(len(tr),device=device)];loss_sum=0.;nb=0
        for st in range(0,len(perm),a.batch_size):
            ix=perm[st:st+a.batch_size]; logits=model(xt[ix]);v=mt[ix].float();el=bce(logits,yt[ix])
            per=(el*v).sum((1,2,3))/v.sum((1,2,3)).clamp_min(1)
            loss=per.mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_sum+=float(loss.detach());nb+=1;updates_done+=1
        sched.step();model.eval()
        with torch.no_grad():
            pv=torch.sigmoid(model(xt[va])).cpu().numpy()[:,0]
        mv=metrics(pv,y[val],valid[val]);mv.update(epoch=ep,train_bce=loss_sum/nb,lr=opt.param_groups[0]['lr'])
        ref=cot[val].astype(np.float64);rv=valid[val]
        rawcot=np.maximum(np.expm1(rlog.astype(np.float64)),0.)
        gated=rawcot*(pv>=.5)
        thick=ref>=30.;raw_thick=rawcot>=30.;gated_thick=gated>=30.
        rcloud=metrics((rawcot>=.1).astype(np.float32),y[val],rv)
        def pod(pred,target,mask):
            tp=np.count_nonzero(pred&target&mask);fn=np.count_nonzero(~pred&target&mask)
            return float(tp/(tp+fn)) if tp+fn else float('nan')
        def csi(pred,target,mask):
            tp=np.count_nonzero(pred&target&mask);fp=np.count_nonzero(pred&~target&mask);fn=np.count_nonzero(~pred&target&mask)
            return float(tp/(tp+fp+fn)) if tp+fp+fn else float('nan')
        mv.update(COT_RMSE_raw=float(np.sqrt(np.mean((rawcot[rv]-ref[rv])**2))),
                  COT_RMSE_gated=float(np.sqrt(np.mean((gated[rv]-ref[rv])**2))),
                  thick_POD_raw=pod(raw_thick,thick,rv),thick_POD_gated=pod(gated_thick,thick,rv),
                  thick_CSI_raw=csi(raw_thick,thick,rv),thick_CSI_gated=csi(gated_thick,thick,rv),
                  R_COT_ge0p1_cloud_proxy=rcloud)
        by_station={}
        for station in ('sili','zhujia'):
            sel=np.asarray([rows[i]['station']==station for i in val])
            ms=metrics(pv[sel],y[val][sel],rv[sel]);
            rcm=metrics((rawcot[sel]>=.1).astype(np.float32),y[val][sel],rv[sel])
            ms.update(COT_RMSE_raw=float(np.sqrt(np.mean((rawcot[sel][rv[sel]]-ref[sel][rv[sel]])**2))),
                      COT_RMSE_gated=float(np.sqrt(np.mean((gated[sel][rv[sel]]-ref[sel][rv[sel]])**2))),
                      thick_POD_raw=pod(raw_thick[sel],thick[sel],rv[sel]),
                      thick_POD_gated=pod(gated_thick[sel],thick[sel],rv[sel]),
                      thick_CSI_raw=csi(raw_thick[sel],thick[sel],rv[sel]),
                      thick_CSI_gated=csi(gated_thick[sel],thick[sel],rv[sel]),
                      R_COT_ge0p1_cloud_proxy=rcm)
            by_station[station]=ms
        mv['by_station']=by_station
        history.append(mv);(a.output/'metrics.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in history))
        score=mv['KSS']
        if score>best:
            best=score;best_epoch=ep
            torch.save({'model':model.state_dict(),'epoch':ep,'seed':a.seed,'validation_KSS':score},a.output/'best.pt')
            np.savez_compressed(a.output/'validation_predictions.npz',prob_cloud=pv.astype(np.float32),
                                y_cloud=y[val].astype(np.uint8),valid=valid[val],sample_indices=val,
                                station=np.asarray([rows[i]['station'] for i in val]),
                                target_time_utc=np.asarray([rows[i]['target_time_utc'] for i in val]))
        if ep in (1,5,10,20,30):
            print(json.dumps({'epoch':ep,'updates':updates_done,'train_bce':mv['train_bce'],'val':mv,'elapsed_sec':time.monotonic()-start}),flush=True)
    if updates_done!=updates: raise RuntimeError(f'update mismatch {updates_done}/{updates}')
    result=dict(state='COMPLETE_CLOUD_GATE_SEED42',best_epoch=best_epoch,best_validation_KSS=best,
                updates=updates_done,expected_updates=updates,elapsed_sec=time.monotonic()-start,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),metrics=history,
                train_label_counts={str(k):int(np.count_nonzero(valid[train]&(clp[train]==k))) for k in (0,1,2)},
                val_label_counts={str(k):int(np.count_nonzero(valid[val]&(clp[val]==k))) for k in (0,1,2)},
                test_payloads_read=0)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
