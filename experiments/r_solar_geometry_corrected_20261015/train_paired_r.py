#!/usr/bin/env python3
"""Same-seed R retrains for old and EOT-corrected geometry on common support."""
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def block(a, b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.SiLU())


class COTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.middle = block(16,32), block(32,64), block(64,128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128,64,2,2), block(128,64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64,32,2,2), block(64,32)
        self.head = nn.Conv2d(32,1,1)
    def forward(self, x):
        a=self.enc1(x); b=self.enc2(F.max_pool2d(a,2)); c=self.middle(F.max_pool2d(b,2))
        d=self.dec2(torch.cat((self.up2(c),b),1))
        return F.softplus(self.head(self.dec1(torch.cat((self.up1(d),a),1))))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def init_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def loss_sum(pred,y,m):
    yy=torch.where(m,y,torch.zeros_like(y)); pp=torch.where(m,pred,torch.zeros_like(pred))
    return F.huber_loss(pp.float(),yy.float(),reduction="none",delta=.1).mul(m).sum()


def update(model,opt,scaler,x,y,m):
    model.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.float16): pred=model(x)
    loss=loss_sum(pred,y,m)/m.sum().clamp_min(1)
    if not torch.isfinite(loss): raise RuntimeError("nonfinite R loss")
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    scaler.step(opt); scaler.update()
    return loss.detach()


@torch.no_grad()
def evaluate(model,x,y,m,ids,batch,save=False):
    model.eval(); total=torch.zeros((),device=x.device); pixels=torch.zeros((),device=x.device); outputs=[]
    for sub in ids.split(batch):
        with torch.autocast("cuda",dtype=torch.float16): pred=model(x[sub]).float()
        if not torch.isfinite(pred).all(): raise RuntimeError("nonfinite R validation prediction")
        total+=loss_sum(pred,y[sub],m[sub]); pixels+=m[sub].sum()
        if save: outputs.append(pred.cpu().numpy())
    if pixels.item()==0: raise RuntimeError("empty validation support")
    return float(total/pixels),(np.concatenate(outputs) if save else None)


def run_arm(pack, arm, out, seed, epochs, batch):
    out.mkdir(parents=True,exist_ok=False)
    xraw=np.load(pack/f"x_{arm}_geometry.npy",mmap_mode="r",allow_pickle=False)
    cot=np.load(pack/"cot.npy",mmap_mode="r",allow_pickle=False)
    mask=np.load(pack/"common_mask.npy",mmap_mode="r",allow_pickle=False).astype(bool)
    rows=list(csv.DictReader((pack/"rows_supervised.csv").open(newline="",encoding="utf-8")))
    if xraw.shape!=(len(rows),16,16,16) or cot.shape!=(len(rows),1,16,16) or mask.shape!=cot.shape:
        raise ValueError("arm arrays do not match supervised row keys")
    tr=np.asarray([i for i,r in enumerate(rows) if r["split"]=="train"],np.int64)
    va=np.asarray([i for i,r in enumerate(rows) if r["split"]=="val"],np.int64)
    if not len(tr) or not len(va): raise ValueError("empty train/validation rows")
    # Same common per-pixel support and same train-only normalization protocol.
    total=np.zeros(16,np.float64); second=np.zeros(16,np.float64); count=np.zeros(16,np.int64)
    for i in tr:
        m=mask[i,0]; xx=np.asarray(xraw[i],np.float32)
        for ch in range(16):
            v=xx[ch][m & np.isfinite(xx[ch])]
            total[ch]+=v.sum(dtype=np.float64); second[ch]+=np.square(v,dtype=np.float64).sum(); count[ch]+=v.size
    if np.any(count==0): raise ValueError(f"empty normalization channel count {count.tolist()}")
    mean=total/count; std=np.sqrt(np.maximum(second/count-mean*mean,1e-8)); mean[15],std[15]=0.,1.
    atomic(out/"normalization.json",dict(mean=mean.tolist(),std=std.tolist(),count=count.tolist(),fit_scope="common train CPP pixels only"))
    x=np.asarray(xraw,np.float32).copy()
    x=(x-mean.astype(np.float32)[None,:,None,None])/std.astype(np.float32)[None,:,None,None]
    x[~np.isfinite(x)]=0.
    y=np.log1p(np.where(mask,cot,0.)).astype(np.float32)
    dev=torch.device("cuda:0"); torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    xt=torch.from_numpy(x).to(dev); yt=torch.from_numpy(y).to(dev); mt=torch.from_numpy(mask).to(dev)
    trt=torch.as_tensor(tr,device=dev); vat=torch.as_tensor(va,device=dev)
    seed_all(seed); model=COTUNet().to(dev); initialization_hash=init_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda")
    meta=dict(experiment="r_solar_geometry_corrected_20261015", arm=arm, seed=seed, initialization="fresh random",
              initialization_state_sha256=initialization_hash, split_id="hunan_monthly_block_2d_buffer1d_seed42_v1",
              input="13 AGRI + cosSOZ + cosRAA + day_mask", geometry=("producer original EOT formula" if arm=="old" else "EOT radians converted to degrees"),
              train_rows=int(len(tr)),val_rows=int(len(va)),train_common_pixels=int(mask[tr].sum()),val_common_pixels=int(mask[va].sum()),
              batch=batch,epochs_max=epochs,optimizer="AdamW lr=1e-3 wd=1e-4",loss="finite-safe masked Huber(log1p(CPP COT), delta=.1)",
              normalization_sha256=sha(out/"normalization.json"),pack_audit_sha256=sha(pack/"pack_audit.json"),
              test_payloads_read=0,gpu=torch.cuda.get_device_name(0),pbs_job_id=os.environ.get("PBS_JOBID"))
    atomic(out/"run_metadata.json",meta)
    best=float("inf"); stale=0; started=time.monotonic()
    for epoch in range(epochs):
        ids=trt[torch.randperm(len(trt),device=dev)]; weighted=0.; pixels=0
        for sub in ids.split(batch):
            v=update(model,opt,scaler,xt[sub],yt[sub],mt[sub]); n=int(mt[sub].sum()); weighted+=float(v)*n; pixels+=n
        val,_=evaluate(model,xt,yt,mt,vat,batch)
        improved=val<best
        if improved: best,stale=val,0
        else: stale+=1
        rec=dict(epoch=epoch,train_huber=weighted/pixels,val_huber=val,best_val_huber=best,elapsed_seconds=time.monotonic()-started)
        with (out/"metrics.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,metadata=meta,record=rec)
        tmp=out/"last.pt.tmp"; torch.save(ck,tmp); tmp.replace(out/"last.pt")
        if improved:
            tmp=out/"best.pt.tmp"; torch.save(ck,tmp); tmp.replace(out/"best.pt")
        atomic(out/"status.json",dict(state="TRAINING",**rec)); print("ARM",arm,"EPOCH",json.dumps(rec),flush=True)
        if stale>=8: break
    ck=torch.load(out/"best.pt",map_location=dev,weights_only=False); model.load_state_dict(ck["model"])
    vl,pred=evaluate(model,xt,yt,mt,vat,batch,save=True)
    np.savez_compressed(out/"validation_predictions.npz",array_indices=va,pred_log1p_cot=pred,
                        reference_cot=np.asarray(cot[va]),mask=np.asarray(mask[va]))
    metrics=dict(validation_huber_log1p=vl,valid_pixels=int(mask[va].sum()),
                 rmse=float(np.sqrt(np.mean((np.expm1(pred.astype(np.float64))[mask[va]]-cot[va][mask[va]])**2))),
                 test_payloads_read=0,interpretation="common CPP-supported daytime pixels; CPP is retrieval reference")
    atomic(out/"validation_metrics.json",metrics)
    status=dict(state="COMPLETE",best_validation_huber=best,last_epoch=epoch,elapsed_seconds=time.monotonic()-started,
                validation_metrics=metrics,test_payloads_read=0)
    atomic(out/"status.json",status)
    atomic(out/"artifact_hashes.json",{n:sha(out/n) for n in ("best.pt","last.pt","normalization.json","run_metadata.json","validation_predictions.npz","validation_metrics.json")})
    print("ARM_COMPLETE",arm,json.dumps(status),flush=True)


@torch.no_grad()
def score_original(pack, checkpoint, normalization, output, batch):
    xraw=np.load(pack/"x_old_geometry.npy",mmap_mode="r",allow_pickle=False)
    cot=np.load(pack/"cot.npy",mmap_mode="r",allow_pickle=False)
    mask=np.load(pack/"common_mask.npy",mmap_mode="r",allow_pickle=False).astype(bool)
    rows=list(csv.DictReader((pack/"rows_supervised.csv").open(newline="",encoding="utf-8")))
    ids=np.asarray([i for i,r in enumerate(rows) if r["split"]=="val"],np.int64)
    norm=json.loads(normalization.read_text(encoding="utf-8")); mean=np.asarray(norm["mean"],np.float32); std=np.asarray(norm["std"],np.float32)
    if mean.shape!=(16,) or std.shape!=(16,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError("invalid original-R normalization")
    x=np.asarray(xraw,np.float32)[ids].copy()
    x=(x-mean[None,:,None,None])/std[None,:,None,None]; x[~np.isfinite(x)]=0.
    dev=torch.device("cuda:0"); xt=torch.from_numpy(x).to(dev)
    model=COTUNet().to(dev); ck=torch.load(checkpoint,map_location=dev,weights_only=False)
    model.load_state_dict(ck["model"]); model.eval(); outpred=[]
    for start in range(0,len(ids),batch):
        with torch.autocast("cuda",dtype=torch.float16): pred=model(xt[start:start+batch]).float()
        if not torch.isfinite(pred).all(): raise RuntimeError("original-R prediction contains nonfinite values")
        outpred.append(pred.cpu().numpy())
    pred=np.concatenate(outpred)
    np.savez_compressed(output/"original_checkpoint_validation_predictions.npz",array_indices=ids,
                        pred_log1p_cot=pred,reference_cot=np.asarray(cot[ids]),mask=np.asarray(mask[ids]))
    atomic(output/"original_checkpoint_evaluation.json",dict(original_checkpoint_sha256=sha(checkpoint),
          original_normalization_sha256=sha(normalization),common_val_rows=len(ids),common_val_pixels=int(mask[ids].sum()),
          prediction_sha256=sha(output/"original_checkpoint_validation_predictions.npz"),
          test_payloads_read=0,interpretation="original frozen R evaluated on same common CPP validation pixels"))
    print("ORIGINAL_CHECKPOINT_EVAL_COMPLETE",(output/"original_checkpoint_evaluation.json").read_text(),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--pack",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--original-checkpoint",type=Path,required=True); p.add_argument("--original-normalization",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch",type=int,default=1284)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    for arm in ("old","corrected"):
        run_arm(a.pack,arm,a.output/arm,a.seed,a.epochs,a.batch)
    score_original(a.pack,a.original_checkpoint,a.original_normalization,a.output,a.batch)


if __name__=="__main__": main()
