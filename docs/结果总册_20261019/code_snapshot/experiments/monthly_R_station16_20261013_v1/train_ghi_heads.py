#!/usr/bin/env python3
"""Matched A/AC/AP station GHI heads on the frozen monthly train/val pack."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from head_candidate_v0 import LocalGHIHead, parameter_counts

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<22),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pack",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--station",choices=("sili","zhujia"),required=True)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=2000)
    p.add_argument("--profile-only",action="store_true");p.add_argument("--profile-output",type=Path)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("formal head training requires CUDA")
    if a.output.exists():raise FileExistsError(a.output)
    meta=json.loads((a.pack/"pack_manifest.json").read_text())
    if meta.get("test_payload_opened") is not False or meta.get("state")!="COMPLETE":raise ValueError("pack manifest gate failed")
    rows=list(csv.DictReader((a.pack/"rows.csv").open(newline="",encoding="utf-8")))
    keep=np.asarray([r["station"]==a.station for r in rows]); train=np.asarray([r["split"]=="train" for r in rows])&keep; val=np.asarray([r["split"]=="val" for r in rows])&keep
    if not train.any() or not val.any():raise ValueError("empty station split")
    device=torch.device("cuda");torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    ag=np.load(a.pack/"agri_raw.npy",mmap_mode="r",allow_pickle=False)
    valid=np.load(a.pack/"agri_valid.npy",mmap_mode="r",allow_pickle=False).astype(bool)
    geo=np.load(a.pack/"geometry.npy",mmap_mode="r",allow_pickle=False)
    cot=np.load(a.pack/"cot_log1p.npy",mmap_mode="r",allow_pickle=False)
    gh=np.asarray([float(r["ghi"]) for r in rows],np.float32);cl=np.asarray([float(r["clear"]) for r in rows],np.float32)
    am=np.asarray(meta["AGRI_norm_mean"],np.float32)[None,:,None,None];ast=np.asarray(meta["AGRI_norm_std"],np.float32)[None,:,None,None]
    cm=np.float32(meta["COT_norm_mean"]);cs=np.float32(meta["COT_norm_std"])
    X=np.where(valid,(np.asarray(ag)-am)/ast,0).astype(np.float32)
    # Geometry is a read-only memmap; own a writable copy before torch.from_numpy.
    G=np.array(geo,dtype=np.float32,copy=True);C=((np.asarray(cot,dtype=np.float32)-cm)/cs).astype(np.float32)
    X=torch.from_numpy(X).to(device);G=torch.from_numpy(G).to(device);C=torch.from_numpy(C).to(device)
    GH=torch.from_numpy(gh).to(device);CL=torch.from_numpy(cl).to(device)
    train_i=np.flatnonzero(train);val_i=np.flatnonzero(val)
    if a.profile_only:
        if a.profile_output is None: raise ValueError("--profile-output is required with --profile-only")
        results=[];limit=min(len(train_i),24155)
        candidates=[b for b in (128,256,512,1024,1536,1792,2000) if b<=limit]
        for arm in ("A","AC","AP"):
            torch.manual_seed(a.seed);m=LocalGHIHead(arm,2).to(device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
            for batch in candidates:
                ix=torch.as_tensor(train_i[:batch],device=device)
                for _ in range(3):
                    out=m(X[ix],G[ix],C[ix] if arm=="AC" else None);loss=((out*CL[ix]-GH[ix])/1000).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();steps=12
                for _ in range(steps):
                    out=m(X[ix],G[ix],C[ix] if arm=="AC" else None);loss=((out*CL[ix]-GH[ix])/1000).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                results.append({"arm":arm,"batch":batch,"examples_per_second":batch*steps/elapsed,"seconds_per_update":elapsed/steps,"peak_GiB":torch.cuda.max_memory_allocated()/1024**3,"updates_per_smallest_site_epoch":24155//batch})
        a.profile_output.parent.mkdir(parents=True,exist_ok=True)
        a.profile_output.write_text(json.dumps({"state":"PROFILE_COMPLETE","results":results,"selection":"largest batch with at least 12 updates per epoch and within 3% of best measured throughput; all three arms profiled","test_payload_opened":False},indent=2)+"\n")
        print(json.dumps(results,indent=2),flush=True);return
    # Common initialization: identical AGRI trunk and common readout weights.
    torch.manual_seed(a.seed)
    base=LocalGHIHead("A",2).to(device)
    base_state={k:v.detach().clone() for k,v in base.state_dict().items()}
    run_meta={"station":a.station,"seed":a.seed,"epochs":a.epochs,"batch_size":a.batch_size,"split_id":meta["split_id"],"pack_manifest_sha256":sha256(a.pack/"pack_manifest.json"),"training_source_sha256":sha256(Path(__file__)),"head_source_sha256":sha256(Path(__file__).with_name("head_candidate_v0.py")),"input_normalization":"train-only AGRI statistics; train-only log1p(COT) statistics; geometry remains physical cosSOZ/cosRAA","loss":"MSE of physical GHI error divided by 1000","test_payload_opened":False,"parameter_counts":parameter_counts(2)}
    a.output.mkdir(parents=True);(a.output/"run_metadata.json").write_text(json.dumps(run_meta,indent=2)+"\n")
    rng=np.random.default_rng(a.seed);logs=[]
    # Freeze one epoch permutation sequence and reuse it across all paired arms.
    epoch_orders=[rng.permutation(train_i) for _ in range(a.epochs)]
    for arm in ("A","AC","AP"):
        torch.manual_seed(a.seed)
        model=LocalGHIHead(arm,2).to(device)
        if arm=="A":model.load_state_dict(base_state)
        else:
            with torch.no_grad():
                model.agri_encoder.load_state_dict(base.agri_encoder.state_dict())
                model.readout[0].weight.zero_();model.readout[0].weight[:,:base.readout[0].in_features].copy_(base.readout[0].weight)
                model.readout[0].bias.copy_(base.readout[0].bias)
                model.readout[2:].load_state_dict({k:v for k,v in base.readout[2:].state_dict().items()})
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5)
        best=float("inf");best_state=None;start=time.time()
        for ep in range(a.epochs):
            model.train();order=epoch_orders[ep];loss_sum=0.;nseen=0
            for st in range(0,len(order),a.batch_size):
                ix=order[st:st+a.batch_size];ti=torch.as_tensor(ix,device=device)
                predkt=model(X[ti],G[ti],C[ti] if arm=="AC" else None)
                err=(predkt*CL[ti]-GH[ti])/1000.
                loss=err.square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                loss_sum+=float(loss.detach())*len(ix);nseen+=len(ix)
            sched.step();model.eval()
            with torch.inference_mode():
                vp=[]
                for st in range(0,len(val_i),a.batch_size):
                    ix=torch.as_tensor(val_i[st:st+a.batch_size],device=device)
                    vk=model(X[ix],G[ix],C[ix] if arm=="AC" else None);vp.append((vk*CL[ix]).cpu().numpy())
                pred=np.concatenate(vp);truth=gh[val_i];rmse=float(np.sqrt(np.mean((pred-truth)**2)));mae=float(np.mean(np.abs(pred-truth)))
            rec={"arm":arm,"epoch":ep,"train_mse_scaled":loss_sum/nseen,"val_ghi_rmse_wm2":rmse,"val_ghi_mae_wm2":mae,"lr":opt.param_groups[0]["lr"]};logs.append(rec)
            if rmse<best:best=rmse;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if ep in (0,a.epochs-1) or (ep+1)%5==0:print(json.dumps(rec),flush=True)
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode():
            pred=[]
            for st in range(0,len(val_i),a.batch_size):
                ix=torch.as_tensor(val_i[st:st+a.batch_size],device=device);pred.append((model(X[ix],G[ix],C[ix] if arm=="AC" else None)*CL[ix]).cpu().numpy())
        pred=np.concatenate(pred).astype(np.float64);truth=gh[val_i].astype(np.float64)
        np.save(a.output/f"{arm}_val_prediction_ghi_wm2.npy",pred)
        torch.save({"model":best_state,"arm":arm,"seed":a.seed,"best_val_rmse_wm2":best},a.output/f"{arm}_best.pt")
        (a.output/f"{arm}_validation.json").write_text(json.dumps({"arm":arm,"station":a.station,"seed":a.seed,"best_val_rmse_wm2":float(np.sqrt(np.mean((pred-truth)**2))),"mae_wm2":float(np.mean(np.abs(pred-truth))),"n":len(val_i),"best_epoch":min((r["epoch"] for r in logs if r["arm"]==arm and r["val_ghi_rmse_wm2"]==best),default=-1),"checkpoint_sha256":sha256(a.output/f"{arm}_best.pt"),"test_payload_opened":False},indent=2)+"\n")
        print(f"ARM_COMPLETE {arm} elapsed_s={time.time()-start:.1f}",flush=True)
    with (a.output/"metrics.jsonl").open("w") as f:
        for r in logs:f.write(json.dumps(r)+"\n")
    print("THREE_ARM_SEED_COMPLETE",flush=True)
if __name__=="__main__":main()
