#!/usr/bin/env python3
"""Fresh monthly-split station-patch AGRI->CPP COT U-Net training."""
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU())


class COTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Match the historical COTUNet16 width/topology (base_ch=32); the
        # monthly contract changes supervision dates, not model capacity.
        self.enc1, self.enc2, self.middle = block(16, 32), block(32, 64), block(64, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat((self.up2(c), b), 1))
        return F.softplus(self.head(self.dec1(torch.cat((self.up1(d), a), 1))))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def loss_sum(pred, target, mask):
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    safe_pred = torch.where(mask, pred, torch.zeros_like(pred))
    return F.huber_loss(safe_pred.float(), safe_target.float(), reduction="none", delta=0.1).mul(mask).sum()


def update(model, opt, scaler, x, y, m):
    model.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model(x)
        loss = loss_sum(pred, y, m) / m.sum().clamp_min(1)
    if not torch.isfinite(loss): raise RuntimeError("nonfinite R training loss")
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    scaler.step(opt); scaler.update()
    return loss.detach()


@torch.no_grad()
def evaluate(model, x, y, m, ids, batch, save=False):
    model.eval(); total = torch.zeros((), device=x.device); pixels = torch.zeros((), device=x.device); outputs=[]
    for sub in ids.split(batch):
        with torch.autocast("cuda", dtype=torch.float16): pred = model(x[sub]).float()
        if not torch.isfinite(pred).all(): raise RuntimeError("nonfinite validation prediction")
        total += loss_sum(pred, y[sub], m[sub]); pixels += m[sub].sum()
        if save: outputs.append(pred.cpu().numpy())
    if pixels.item() == 0: raise RuntimeError("validation has no valid reference pixels")
    return float(total / pixels), (np.concatenate(outputs) if save else None)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--pack",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=50)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    pack_status=json.loads((a.pack/"pack_status.json").read_text())
    if pack_status.get("state")!="COMPLETE_MONTHLY_R_STATION16_PACK" or pack_status.get("test_payloads_read")!=0:
        raise RuntimeError("pack is incomplete or test was read")
    with (a.pack/"rows.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if any(r["split"] not in ("train","val") for r in rows): raise RuntimeError("unexpected split in rows")
    sup=[r for r in rows if r["state"]=="SUPERVISED"]
    if any(int(r["supervised_pixels"])<=0 for r in sup): raise RuntimeError("bad supervised row")
    train_rows=[i for i,r in enumerate(sup) if r["split"]=="train"]
    val_rows=[i for i,r in enumerate(sup) if r["split"]=="val"]
    if not train_rows or not val_rows: raise RuntimeError("empty train or val support")
    xraw=np.load(a.pack/"x_raw.npy",mmap_mode="r",allow_pickle=False)
    cot=np.load(a.pack/"cot.npy",mmap_mode="r",allow_pickle=False)
    mask=np.load(a.pack/"mask.npy",mmap_mode="r",allow_pickle=False)
    if xraw.shape!=(len(sup),16,16,16) or cot.shape!=(len(sup),1,16,16) or mask.shape!=(len(sup),1,16,16):
        raise ValueError(f"array/row mismatch: {xraw.shape} {cot.shape} {mask.shape} {len(sup)}")
    # Train-only channel normalization, accumulated over the CPP+input-valid support.
    total=np.zeros(16,np.float64); second=np.zeros(16,np.float64); count=np.zeros(16,np.int64)
    for ix in train_rows:
        m=np.asarray(mask[ix,0],bool); x=np.asarray(xraw[ix],np.float32)
        for ch in range(16):
            v=x[ch][m & np.isfinite(x[ch])]
            total[ch]+=v.sum(dtype=np.float64); second[ch]+=np.square(v,dtype=np.float64).sum(); count[ch]+=v.size
    if np.any(count==0): raise RuntimeError(f"empty train normalization channels: {count.tolist()}")
    mean=total/count; std=np.sqrt(np.maximum(second/count-mean*mean,1e-8)); mean[15],std[15]=0.,1.
    atomic(a.output/"normalization.json",dict(mean=mean.tolist(),std=std.tolist(),count=count.tolist(),fit_scope="train samples and train supervision mask only"))
    # Make finite-safe normalized CPU buffers then place on one GPU for throughput profiling and fit.
    x=np.asarray(xraw,np.float32).copy();
    x=(x-mean.astype(np.float32)[None,:,None,None])/std.astype(np.float32)[None,:,None,None]
    x[~np.isfinite(x)]=0.
    cot=np.asarray(cot,np.float32).copy(); mask=np.asarray(mask,bool).copy()
    if np.any(mask & (~np.isfinite(cot) | (cot<0) | (cot>100))): raise RuntimeError("invalid CPP label under mask")
    y=np.log1p(np.where(mask,cot,0.)).astype(np.float32)
    device=torch.device("cuda:0")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True; seed_all(a.seed)
    xt=torch.from_numpy(x).to(device); yt=torch.from_numpy(y).to(device); mt=torch.from_numpy(mask).to(device)
    tr=torch.as_tensor(train_rows,device=device); va=torch.as_tensor(val_rows,device=device)
    # Keep at least 12 optimizer updates per epoch, then profile larger
    # candidates up to that data-limited cap. This avoids fake GPU occupancy
    # from one giant, low-update batch on a small 16x16 task.
    batch_cap=max(1,len(train_rows)//12)
    candidates=tuple(sorted({min(v,batch_cap) for v in (128,256,512,1024,2048,4096)}))
    profile=[]
    for b in candidates:
        seed_all(a.seed); model=COTUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda")
        ix=tr[:min(b,len(tr))]; torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
        try:
            for _ in range(3): update(model,opt,scaler,xt[ix],yt[ix],mt[ix])
            torch.cuda.synchronize(); start=time.monotonic()
            for _ in range(10): update(model,opt,scaler,xt[ix],yt[ix],mt[ix])
            torch.cuda.synchronize(); elapsed=time.monotonic()-start
            profile.append(dict(batch=int(len(ix)),seconds_per_update=elapsed/10,examples_per_second=len(ix)*10/elapsed,
                                peak_GiB=torch.cuda.max_memory_allocated()/2**30))
        except torch.cuda.OutOfMemoryError:
            profile.append(dict(batch=b,error="OOM")); torch.cuda.empty_cache()
        del model,opt,scaler
    valid=[q for q in profile if "error" not in q]
    if not valid: raise RuntimeError("no batch size fits")
    fastest=max(q["examples_per_second"] for q in valid)
    eligible=[q for q in valid if q["examples_per_second"]>=.97*fastest and q["peak_GiB"] < torch.cuda.get_device_properties(0).total_memory/2**30*.85]
    if not eligible: eligible=valid
    batch=max(q["batch"] for q in eligible)
    atomic(a.output/"profile.json",dict(candidates=profile,selected_batch=batch,
           updates_per_epoch=math.ceil(len(train_rows)/batch),
           selection="largest profiled batch within 3% of best throughput, below 85% memory, and at least 12 updates/epoch when data size allows"))
    seed_all(a.seed); model=COTUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda")
    meta=dict(model="monthly_station16_COTUNet_seed42_v1",split_id="hunan_monthly_block_2d_buffer1d_seed42_v1",
              initialization="random",seed=a.seed,input_features=[f"AGRI_channel_{i}" for i in (0,1,2,3,4,5,8,9,10,11,12,13,14)]+["cosSOZ","cosRAA","day_mask"],
              crop="station-centered 16x16 around Sili and Zhujia",base_channels=32,
              target="log1p(CPP COT), valid [0,100]",
              mask=pack_status["mask"],loss="finite-safe masked Huber(log1p(COT), delta=0.1)",optimizer="AdamW(lr=1e-3, weight_decay=1e-4)",
              epochs_max=a.epochs,train_examples=len(train_rows),val_examples=len(val_rows),normalization_sha256=sha(a.output/"normalization.json"),
              pack_status_sha256=sha(a.pack/"pack_status.json"),pack_arrays={n:sha(a.pack/n) for n in ("x_raw.npy","cot.npy","mask.npy","rows.csv")},
              training_script_sha256=sha(Path(__file__).resolve()),
              train_rows_sha256=hashlib.sha256("\n".join(str(i) for i in train_rows).encode()).hexdigest(),
              val_rows_sha256=hashlib.sha256("\n".join(str(i) for i in val_rows).encode()).hexdigest(),
              test_payloads_read=0,gpu=torch.cuda.get_device_name(0),pbs_job_id=os.environ.get("PBS_JOBID"),batch=batch)
    atomic(a.output/"run_metadata.json",meta)
    best=float("inf"); stale=0; started=time.monotonic()
    for epoch in range(a.epochs):
        ids=tr[torch.randperm(len(tr),device=device)]; s=0.; n=0
        for sub in ids.split(batch):
            loss=update(model,opt,scaler,xt[sub],yt[sub],mt[sub]); pix=int(mt[sub].sum()); s+=float(loss)*pix; n+=pix
        vl,_=evaluate(model,xt,yt,mt,va,batch)
        improved=vl<best
        if improved: best,stale=vl,0
        else: stale+=1
        record=dict(epoch=epoch,train_huber=s/n,validation_huber=vl,best_validation_huber=best,elapsed_seconds=time.monotonic()-started)
        with (a.output/"metrics.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
        ckpt=dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,metadata=meta,record=record)
        for name in (["best.pt","last.pt"] if improved else ["last.pt"]):
            tmp=a.output/(name+".tmp"); torch.save(ckpt,tmp); tmp.replace(a.output/name)
        atomic(a.output/"status.json",dict(state="TRAINING",**record)); print("EPOCH",json.dumps(record),flush=True)
        if stale>=8: break
    ckpt=torch.load(a.output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ckpt["model"])
    val_loss,pred=evaluate(model,xt,yt,mt,va,batch,save=True)
    np.savez_compressed(a.output/"validation_predictions.npz",sample_indices=np.asarray(val_rows),pred_log1p_cot=pred,
                        reference_cot=cot[val_rows],mask=mask[val_rows])
    pcot=np.expm1(pred.astype(np.float64)); ref=cot[val_rows].astype(np.float64); vm=mask[val_rows]
    err=(pcot-ref)[vm]
    metrics=dict(validation_huber_log1p=val_loss,cot_rmse=float(np.sqrt(np.mean(err**2))),cot_mae=float(np.mean(np.abs(err))),
                 valid_pixels=int(vm.sum()),thick_ge30_pixels=int(((ref>=30)&vm).sum()),nearzero_pixels=int(((ref<.1)&vm).sum()),
                 test_payloads_read=0,interpretation="CPP-supported daytime retrieval-reference validation; not independent truth")
    atomic(a.output/"validation_metrics.json",metrics)
    hashes={n:sha(a.output/n) for n in ("best.pt","last.pt","run_metadata.json","normalization.json","validation_predictions.npz","validation_metrics.json","profile.json")}
    atomic(a.output/"artifact_sha256.json",hashes)
    atomic(a.output/"status.json",dict(state="COMPLETE",best_validation_huber=best,last_epoch=epoch,test_payloads_read=0,elapsed_seconds=time.monotonic()-started))
    print("MONTHLY_R_TRAIN_COMPLETE",json.dumps(metrics),flush=True)


if __name__=="__main__": main()
