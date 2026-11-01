"""One-factor COT-aware fine-tuning of the existing Hunan SimVP."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, random, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cot_auxiliary import cpp_only_cot_target, frozen_r_cot_loss, station_r_inputs

NSTEP, BATCH, SEED = 4000, 8, 42
SPLIT = "hunan_monthly_block_2d_buffer1d_seed42_v1"
SIMVP_CKPT_SHA = "f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4"
SIMVP_NORM_SHA = "6290d3ebad884ccfe689f63cc09ab66ab4c3906665c3149995f75c201de64d1d"
SIMVP_MODEL_SHA = "9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb"
SIMVP_CODE_SHA = "2e0d280f7962a77f0c9b7e4cc817fed21d142c1436a12eb858e3234485313d16"
R_CKPT_SHA = "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001"
R_SOURCE_SHA = "7a2e5cb441cdb5132a537cd03580017801595bdea839719d1773398d267210ce"
FRAME_KEYS_SHA = "75fbd1c17cd0a1ddd7c645ea78b3718e045b7fdafbc2ffdf970a34679e623727"
R_NORM_SHA = "a3e68aabd61be43c4f6040c2bd142d1b21ee816e2fe46a9de91e9eaf931328a7"
R_BANK_COT_SHA = "4994c809bdc58fc9f08664072f282b7dd5cc9bdcb69e9a791cddd9a8225963a1"
CPP_COT_SHA = "2791e21aea1ace67fa43c69520bcb3732092d37fc8c2fb866438c9a00e56cb65"
CPP_MASK_SHA = "6659a170a6c087ffe56962d4362117b93ffed5bc2bcf2659320a00a87299d3c5"
GEO_SHA = "8e31b06659374f609dd607560543cc6364b5069139023693c806ddeafdd41c92"
FRAME_VALID_SHA = "b04a8a62c2ac266ddcb7d7ef1f949a607f3f82c6045dcf1cacd80e929df23a6e"


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()


def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(p)


def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)


class Indexed(Dataset):
    def __init__(self,base): self.base=base
    def __len__(self): return len(self.base)
    def __getitem__(self,i): return (*self.base[i],int(i))


class Sidecars:
    def __init__(self,pack,side,rbank,split,dataset):
        self.pack,self.side,self.split,self.rbank=pack,side,split,rbank
        pm=json.loads((pack/"pack_manifest.json").read_text())
        sm=json.loads((side/"pack_manifest.json").read_text())
        if pm.get("state")!="PASS_TRAIN_VAL_SEQUENCE_CPP_PACK" or sm.get("state")!="COMPLETE":
            raise ValueError("train/val sidecar pack is incomplete")
        if pm.get("test_payload_opened") is not False or sm.get("test_payload_opened") is not False:
            raise ValueError("test-payload exclusion not certified")
        if pm["source_sha256"]["frame_keys.csv"]!=FRAME_KEYS_SHA or sm["bank_frame_keys_sha256"]!=FRAME_KEYS_SHA:
            raise ValueError("COT sidecars are not bound to the frozen frame keys")
        expected_pack={"cpp_cot_physical.npy":CPP_COT_SHA,"cpp_common_mask.npy":CPP_MASK_SHA}
        for name,expected in expected_pack.items():
            if sha(pack/name)!=expected: raise ValueError("CPP array checksum mismatch: "+name)
        expected_side={"frame_geometry.npy":GEO_SHA,"frame_valid_mask.npy":FRAME_VALID_SHA}
        for name,expected in expected_side.items():
            if sha(side/name)!=expected: raise ValueError("frame sidecar checksum mismatch: "+name)
        if sha(rbank/"cot_log1p.npy")!=R_BANK_COT_SHA or sha(rbank/"frame_keys.csv")!=FRAME_KEYS_SHA:
            raise ValueError("R COT bank checksum mismatch")
        self.seq=np.load(pack/f"sequence_indices_{split}.npy",mmap_mode="r",allow_pickle=False)
        if self.seq.shape!=(len(dataset),24): raise ValueError("sequence/sample count mismatch")
        seqkeys=[r for r in csv.DictReader((pack/"sequence_keys.csv").open(newline="",encoding="utf-8")) if r["split"]==split]
        framekeys=list(csv.DictReader((rbank/"frame_keys.csv").open(newline="",encoding="utf-8")))
        if len(seqkeys)!=len(dataset) or len(framekeys)!=55733: raise ValueError("sequence or frame key count mismatch")
        for i,(k,row) in enumerate(zip(seqkeys,dataset.rows)):
            if int(k["split_index"])!=i: raise ValueError("split index is not canonical")
            start=datetime.strptime(row["BJT_start"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=8)))
            if (start+timedelta(minutes=105)).isoformat(timespec="minutes")!=k["init_time_bjt"]:
                raise ValueError(f"init-time join failed at {split}/{i}")
            paths=row["data_relpaths"].split("|")
            for j in range(8,24):
                # The sample manifest's BJT_start is BJT, but this preprocessed
                # AGRI path naming is UTC. Do not apply the matched-NPZ rule here.
                utc=datetime.strptime(Path(paths[j]).stem,"%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
                fk=framekeys[int(self.seq[i,j])]
                if fk["split"]!=split or fk["target_time_utc"]!=utc:
                    raise ValueError(f"target-frame join failed at {split}/{i}/{j-7}")
        self.cpp=np.load(pack/"cpp_cot_physical.npy",mmap_mode="r",allow_pickle=False)
        self.cm=np.load(pack/"cpp_common_mask.npy",mmap_mode="r",allow_pickle=False)
        self.geo=np.load(side/"frame_geometry.npy",mmap_mode="r",allow_pickle=False)
        self.valid=np.load(side/"frame_valid_mask.npy",mmap_mode="r",allow_pickle=False)
        self.rlog=np.load(rbank/"cot_log1p.npy",mmap_mode="r",allow_pickle=False)
        self.centers=np.zeros((len(self.rlog),2,2),np.int16)
        for r in csv.DictReader((rbank/"frame_keys.csv").open(newline="",encoding="utf-8")):
            i=int(r["index"]);self.centers[i,0]=np.fromstring(r["sili_grid_rc"],dtype=int,sep=",")
            self.centers[i,1]=np.fromstring(r["zhujia_grid_rc"],dtype=int,sep=",")
        if self.cpp.shape!=(55733,2,1,16,16) or self.cpp.shape!=self.cm.shape or self.cpp.shape!=self.valid.shape:
            raise ValueError("CPP/R mask dimensions differ")
        if self.geo.shape!=(55733,2,3,16,16) or self.rlog.shape!=self.cpp.shape:
            raise ValueError("R geometry or target bank dimensions differ")

    def batch(self,ids,device):
        frames=np.asarray(self.seq[np.asarray(ids,dtype=np.int64),8:24],dtype=np.int64)
        def take(a): return torch.from_numpy(np.asarray(a[frames]).copy()).to(device,non_blocking=True)
        return dict(cpp=take(self.cpp),cpp_mask=take(self.cm).bool(),geo=take(self.geo),
                    valid=take(self.valid).bool(),rlog=take(self.rlog),
                    centers=torch.from_numpy(self.centers[frames].copy()).to(device,non_blocking=True))


def score_validation_cot(net,rnet,val,side,mean,std,rmean,rstd,output,workers):
    """Save a key-preserving validation COT bank and FP64 same-mask metrics."""
    output.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(Indexed(val),batch_size=BATCH,shuffle=False,num_workers=workers,
        pin_memory=True,persistent_workers=workers>0,prefetch_factor=1 if workers>0 else None)
    banks=[];seqids=[]
    sse=np.zeros((2,16),np.float64);count=np.zeros((2,16),np.int64)
    sse3=np.zeros_like(sse);count3=np.zeros_like(count)
    tp=np.zeros((2,16),np.int64);fp=tp.copy();fn=tp.copy()
    near_support=np.zeros((2,16),np.int64);near_false=np.zeros((2,16),np.int64)
    mae_sum=np.zeros((2,16),np.float64);bias_sum=np.zeros((2,16),np.float64)
    net.eval()
    with torch.no_grad():
        for x,g,y,v,month,ids in loader:
            x,g=[q.cuda(non_blocking=True) for q in (x,g)]
            with torch.autocast("cuda",dtype=torch.bfloat16): pred=net(x,g).float()
            pack=side.batch(ids.numpy().tolist(),pred.device)
            ri=station_r_inputs(pred,pack["geo"],pack["centers"],mean,std,rmean,rstd)
            rhat=rnet(ri.reshape(-1,16,16,16)).float().reshape(len(ids),16,2,1,16,16)
            # COTUNet emits log1p(COT); retain that contract in the saved bank.
            logcot=rhat.permute(0,2,1,3,4,5).squeeze(3).cpu().numpy().astype(np.float32)
            banks.append(logcot);seqids.extend(ids.tolist())
            ycot=pack["cpp"].float().cpu().numpy()[:,:, :,0]
            mask=(pack["cpp_mask"]&pack["valid"]).cpu().numpy()[:,:,:,0]
            pcot=np.expm1(logcot.astype(np.float64))
            for si in range(2):
                for ti in range(16):
                    m=mask[:,ti,si]
                    if not m.any(): continue
                    yt=ycot[:,ti,si][m].astype(np.float64);yp=pcot[:,si,ti][m]
                    sse[si,ti]+=np.square(yp-yt).sum(dtype=np.float64);count[si,ti]+=m.sum()
                    mae_sum[si,ti]+=np.abs(yp-yt).sum(dtype=np.float64);bias_sum[si,ti]+=(yp-yt).sum(dtype=np.float64)
                    pth=yp>=30;yth=yt>=30
                    tp[si,ti]+=np.count_nonzero(pth&yth);fp[si,ti]+=np.count_nonzero(pth&~yth);fn[si,ti]+=np.count_nonzero(~pth&yth)
                    low=yt<1
                    near_support[si,ti]+=int(low.sum());near_false[si,ti]+=int(np.count_nonzero((yp>=1)&low))
                    # In a 16x16 patch centered on the station pixel at (8,8),
                    # the aligned 3x3 readout is rows/cols 7:10.
                    ml=mask[:,ti,si,7:10,7:10]
                    yl=ycot[:,ti,si,7:10,7:10][ml].astype(np.float64)
                    pl=pcot[:,si,ti,7:10,7:10][ml]
                    if ml.any():
                        sse3[si,ti]+=np.square(pl-yl).sum(dtype=np.float64);count3[si,ti]+=ml.sum()
    bank=np.concatenate(banks,axis=0)
    if bank.shape!=(len(val),2,16,16,16) or len(seqids)!=len(val): raise ValueError("validation COT bank shape/key mismatch")
    np.save(output/"forecast_log1p_cot.npy",bank,allow_pickle=False)
    np.save(output/"validation_dataset_indices.npy",np.asarray(seqids,dtype=np.int32),allow_pickle=False)
    leadrmse=np.sqrt(np.divide(sse,count,out=np.full_like(sse,np.nan),where=count>0))
    leadrmse3=np.sqrt(np.divide(sse3,count3,out=np.full_like(sse3,np.nan),where=count3>0))
    macro=float(np.nanmean(leadrmse));macro3=float(np.nanmean(leadrmse3))
    csi=np.divide(tp,tp+fp+fn,out=np.full_like(tp,np.nan,dtype=np.float64),where=(tp+fp+fn)>0)
    pooled=float(tp.sum()/(tp.sum()+fp.sum()+fn.sum())) if (tp+fp+fn).sum() else float("nan")
    near_far=np.divide(near_false,near_support,out=np.full_like(near_false,np.nan,dtype=np.float64),where=near_support>0)
    mean_mae=np.divide(mae_sum,count,out=np.full_like(mae_sum,np.nan),where=count>0)
    mean_bias=np.divide(bias_sum,count,out=np.full_like(bias_sum,np.nan),where=count>0)
    atomic(output/"cot_aux_validation_score.json",{"status":"PASS_CPP_COMMON_SUPPORT_FP64",
        "CPP_reference":"retrieval product, not independent truth","split_id":SPLIT,
        "validation_sequences":len(val),"valid_cpp_dates":46,"test_payloads_read":0,
        "physical_cot_units":True,"station_lead_macro_rmse_full_patch":macro,
        "station_lead_macro_rmse_center3x3":macro3,
        "station_lead_rmse":leadrmse.tolist(),"station_lead_valid_pixels":count.tolist(),
        "center3x3_station_lead_rmse":leadrmse3.tolist(),"center3x3_valid_pixels":count3.tolist(),
        "station_lead_mae_full_patch":mean_mae.tolist(),"station_lead_bias_full_patch":mean_bias.tolist(),
        "thick_csi_by_station_lead":csi.tolist(),"pooled_thick_csi":pooled,
        "near_zero_false_alarm_rate_cot1":near_far.tolist(),"near_zero_support":near_support.tolist(),
        "bank_layout":"[validation_sequence, station(sili,zhujia), lead(+15..+240), 16,16], stores log1p(COT)",
        "bank_sha256":sha(output/"forecast_log1p_cot.npy"),
        "validation_indices_sha256":sha(output/"validation_dataset_indices.npy")})


def main():
    p=argparse.ArgumentParser()
    for name in ("project","source","agri-root","cache","manifests","checkpoint","r-root","r-bank","cpp-pack","sidecars","control","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA is required")
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;seed_all()
    basecode=a.project/"experiments/hunan_simvp_supervision_20261003_v2/code"
    sys.path[:0]=[str(basecode),str(a.r_root/"scripts")]
    from train_simvp_supervision_20261003 import load_contract,base_mse,eval_model
    from train_station16_R import COTUNet
    ds,val,norm,baseckpt,net=load_contract(SimpleNamespace(source=a.source,root=a.agri_root,
        cache=a.cache,manifests=a.manifests,checkpoint=a.checkpoint))
    rs=a.r_root/"scripts/train_station16_R.py";rc=a.r_root/"results/seed42_v1/best.pt"
    rn=a.r_root/"results/seed42_v1/normalization.json"
    checks={str(a.checkpoint):SIMVP_CKPT_SHA,str(a.manifests/"hunan_train_norm.json"):SIMVP_NORM_SHA,
        str(a.source/"triad_models.py"):SIMVP_MODEL_SHA,str(a.source/"simvp.py"):SIMVP_CODE_SHA,
        str(rs):R_SOURCE_SHA,str(rc):R_CKPT_SHA,str(rn):R_NORM_SHA}
    for path,expected in checks.items():
        if sha(Path(path))!=expected: raise ValueError("source/checkpoint SHA mismatch: "+path)
    net.cuda()
    rnorm=json.loads(rn.read_text());rp=torch.load(rc,map_location="cpu",weights_only=False)
    rnet=COTUNet().float().eval().requires_grad_(False).cuda();rnet.load_state_dict(rp["model"],strict=True)
    rmean=torch.tensor(rnorm["mean"],dtype=torch.float32,device="cuda");rstd=torch.tensor(rnorm["std"],dtype=torch.float32,device="cuda")
    tr=Sidecars(a.cpp_pack,a.sidecars,a.r_bank,"train",ds)
    va=Sidecars(a.cpp_pack,a.sidecars,a.r_bank,"val",val)
    control=json.loads((a.control/"fixed_training_rows.json").read_text())
    cm=json.loads((a.control/"mse/run_metadata.json").read_text());order=control["rows"]
    order_sha=hashlib.sha256(np.asarray(order,dtype=np.int32).tobytes()).hexdigest()
    if cm["steps"]!=NSTEP or cm["effective_batch"]!=BATCH or cm["seed"]!=SEED or order_sha!=control["sha256"] or len(order)!=BATCH*NSTEP:
        raise ValueError("equal-budget MSE row/seed/batch contract mismatch")
    mse_checkpoint=a.control/"mse/step_4000.pt"
    if not mse_checkpoint.is_file(): raise FileNotFoundError("equal-budget MSE step_4000 checkpoint is missing")
    mse_payload=torch.load(mse_checkpoint,map_location="cpu",weights_only=False)
    if mse_payload.get("step")!=NSTEP: raise ValueError("equal-budget MSE checkpoint is not step 4000")
    mse_checkpoint_sha=sha(mse_checkpoint)
    mean=torch.tensor(norm["mean"],dtype=torch.float32,device="cuda")
    std=torch.tensor(norm["std"],dtype=torch.float32,device="cuda")
    # Train-only 32-window gradient-ratio calibration, round-robin by month.
    buckets={m:[] for m in range(1,13)}
    for i,row in enumerate(ds.rows): buckets[int(row["BJT_start"][5:7])].append(i)
    rng=np.random.default_rng(SEED)
    for v in buckets.values(): rng.shuffle(v)
    cal=[]
    while len(cal)<32:
        for m in range(1,13):
            if buckets[m] and len(cal)<32: cal.append(buckets[m].pop())
    calrec=[]
    for st in range(0,32,BATCH):
        ids=cal[st:st+BATCH];samples=[ds[i] for i in ids]
        x,g,y,v=[torch.from_numpy(np.stack([z[j] for z in samples])).cuda() for j in range(4)]
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): pred=net(x,g).float()
        pred=pred.detach().requires_grad_(True);side=tr.batch(ids,pred.device)
        ri=station_r_inputs(pred,side["geo"],side["centers"],mean,std,rmean,rstd)
        target,mask,_,_=cpp_only_cot_target(side["cpp"],side["cpp_mask"],side["valid"])
        lc,_=frozen_r_cot_loss(rnet,ri,target,mask);la=base_mse(pred,y,v)
        ga=torch.autograd.grad(la,pred,retain_graph=True)[0];gc=torch.autograd.grad(lc,pred)[0]
        ratio=(ga.float().norm()/gc.float().norm().clamp_min(1e-12)).item()
        calrec.append({"rows":ids,"L_AGRI":float(la),"L_COT":float(lc),"gradient_ratio":ratio})
    lam=.1*float(np.median([r["gradient_ratio"] for r in calrec]))
    if not math.isfinite(lam) or lam<=0: raise ValueError("invalid train-only COT loss weight")
    a.output.mkdir(parents=True)
    atomic(a.output/"preflight.json",{"status":"PASS","lambda_cot":lam,"calibration":calrec,
        "fixed_rows_sha256":control["sha256"],"test_payloads_read":0,"R_frozen":True})
    # Real-data profile, discarded before the formal updates.
    wrapped=Indexed(ds);loader=DataLoader(wrapped,batch_size=BATCH,sampler=order[:BATCH*4],
        num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,
        prefetch_factor=1 if a.workers>0 else None)
    net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4);profile=[]
    torch.cuda.reset_peak_memory_stats()
    for x,g,y,v,month,ids in loader:
        x,g,y,v=[q.cuda(non_blocking=True) for q in (x,g,y,v)];ids=ids.numpy().tolist()
        torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=net(x,g).float();side=tr.batch(ids,pred.device)
            ri=station_r_inputs(pred,side["geo"],side["centers"],mean,std,rmean,rstd)
            target,mask,_,_=cpp_only_cot_target(side["cpp"],side["cpp_mask"],side["valid"])
            lc,_=frozen_r_cot_loss(rnet,ri,target,mask);la=base_mse(pred,y,v);loss=la+lam*lc
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();profile.append({"seconds":time.monotonic()-start,"L_AGRI":float(la),"L_COT":float(lc),"grad":float(gn)})
    peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
    atomic(a.output/"profile.json",{"status":"PASS" if peak<.85*total else "FAIL","discarded_steps":4,
        "peak_GiB":peak/2**30,"device":torch.cuda.get_device_name(0),"total_GiB":total/2**30,
        "samples_per_second":BATCH/float(np.mean([z["seconds"] for z in profile[1:]])),"records":profile})
    if peak>=.85*total: raise RuntimeError("profile exceeded memory gate")
    net.load_state_dict(baseckpt["model"],strict=True);seed_all();net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-5,weight_decay=1e-4)
    meta={"arm":"AGRI_MSE_plus_CPP_only_COT_aux","cot_target_source":"CPP_only_no_R_fallback",
        "split_id":SPLIT,"seed":SEED,"steps":NSTEP,"batch":BATCH,
        "initialization_sha256":SIMVP_CKPT_SHA,"R_checkpoint_sha256":R_CKPT_SHA,"R_source_sha256":R_SOURCE_SHA,
        "lambda_cot":lam,"fixed_training_rows_sha256":control["sha256"],
        "equal_budget_mse_step4000_sha256":mse_checkpoint_sha,"test_payloads_read":0}
    atomic(a.output/"run_metadata.json",meta)
    loader=DataLoader(Indexed(ds),batch_size=BATCH,sampler=order,num_workers=a.workers,pin_memory=True,
        persistent_workers=a.workers>0,prefetch_factor=1 if a.workers>0 else None,
        generator=torch.Generator().manual_seed(SEED))
    val0=eval_model(net,val,batch_size=BATCH,workers=a.workers);val0["step"]=0
    atomic(a.output/"validation_step_0000.json",val0)
    tmp=a.output/"step_0000.pt.tmp";torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"metadata":meta,"step":0},tmp);tmp.replace(a.output/"step_0000.pt")
    net.train();step=0;records=[];started=time.monotonic()
    for x,g,y,v,month,ids in loader:
        x,g,y,v=[q.cuda(non_blocking=True) for q in (x,g,y,v)];ids=ids.numpy().tolist()
        lr=1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*step/NSTEP))
        for group in opt.param_groups: group["lr"]=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=net(x,g).float();side=tr.batch(ids,pred.device)
            ri=station_r_inputs(pred,side["geo"],side["centers"],mean,std,rmean,rstd)
            target,mask,_,_=cpp_only_cot_target(side["cpp"],side["cpp_mask"],side["valid"])
            lc,_=frozen_r_cot_loss(rnet,ri,target,mask);la=base_mse(pred,y,v);loss=la+lam*lc
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite total loss")
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1
        records.append({"step":step,"L_AGRI":float(la.detach()),"L_COT":float(lc.detach()),"lr":lr,"grad":float(gn)})
        if step%100==0: atomic(a.output/"recent_steps.json",{"step":step,"last100":records[-100:]})
        if step in (2000,4000):
            net.eval();metrics=eval_model(net,val,batch_size=BATCH,workers=a.workers);metrics["step"]=step
            atomic(a.output/f"validation_step_{step:04d}.json",metrics)
            tmp=a.output/f"step_{step:04d}.pt.tmp";torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),"metadata":meta,"step":step},tmp);tmp.replace(a.output/f"step_{step:04d}.pt");net.train()
        if step==NSTEP: break
    if step!=NSTEP: raise RuntimeError(f"expected {NSTEP} steps, got {step}")
    # Score all three frozen references on the exact same CPP support and save
    # their key-preserving COT banks for independent paired recomputation.
    score_validation_cot(net,rnet,val,va,mean,std,rmean,rstd,a.output/"cot_score_aux_4000",a.workers)
    net.load_state_dict(baseckpt["model"],strict=True)
    score_validation_cot(net,rnet,val,va,mean,std,rmean,rstd,a.output/"cot_score_original_simvp",a.workers)
    net.load_state_dict(mse_payload["model"],strict=True)
    score_validation_cot(net,rnet,val,va,mean,std,rmean,rstd,a.output/"cot_score_mse_4000",a.workers)
    atomic(a.output/"training_complete.json",{"status":"PASS_TRAIN_AND_COT_SCORE","steps":step,
        "elapsed_seconds":time.monotonic()-started,"test_payloads_read":0,
        "cot_scores":["cot_score_aux_4000/cot_aux_validation_score.json",
                       "cot_score_original_simvp/cot_aux_validation_score.json",
                       "cot_score_mse_4000/cot_aux_validation_score.json"]})
    print("COT_AUX_TRAIN_COMPLETE",json.dumps({"steps":step,"lambda_cot":lam}),flush=True)


if __name__=="__main__": main()
