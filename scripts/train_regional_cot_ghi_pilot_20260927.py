"""Matched zero/forecast/oracle COT GHI pilots on spatial regional/local grids.

All arms share forecast-only image features and exactly the same GHI keys.
Oracle is a nondeployable diagnostic. No test split is accepted.
"""
import argparse
import math
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from build_regional_cot_ghi_pilot_20260927 import sha


class CausalConv(nn.Module):
    def __init__(self, cin, cout, dilation=1, spatial_stride=2):
        super().__init__()
        self.d = dilation
        self.conv = nn.Conv3d(cin, cout, 3, stride=(1,spatial_stride,spatial_stride), dilation=(dilation,1,1))

    def forward(self, x):
        return F.silu(self.conv(F.pad(x, (1,1,1,1,2*self.d,0))))


class RegionalStationHead(nn.Module):
    def __init__(self):
        super().__init__()
        def image_encoder(size):
            return nn.Sequential(nn.Conv2d(16,16,3,2,1), nn.SiLU(),
                nn.Conv2d(16,16,3,2,1), nn.SiLU(), nn.Flatten(), nn.Linear(16*(size//4)**2,64), nn.SiLU())
        self.local_image = image_encoder(64)
        self.region_image = image_encoder(32)
        self.local_cot = nn.Sequential(CausalConv(2,8), CausalConv(8,8,2),
                                       CausalConv(8,8,4,1), CausalConv(8,8,8,1))
        self.region_cot = nn.Sequential(CausalConv(2,8), CausalConv(8,8,2),
                                        CausalConv(8,8,4,1), CausalConv(8,8,8,1))
        # Flattened 2-D feature grids keep positions; there is no global COT pooling.
        self.local_read = nn.Sequential(nn.Linear(8*16*16,64), nn.SiLU())
        self.region_read = nn.Sequential(nn.Linear(8*8*8,64), nn.SiLU())
        self.station = nn.Embedding(2,8)
        self.lead = nn.Embedding(16,8)
        self.readout = nn.Sequential(nn.Linear(64*4+16,128), nn.SiLU(), nn.Dropout(.1),
                                    nn.Linear(128,1), nn.Softplus())

    def forward(self, local_image, region_image, local_cot, region_cot, station, lead):
        prefix = torch.arange(24, device=lead.device)[None] <= (8+lead[:,None])
        def encode(cot, encoder, read):
            cot = cot * prefix[:, :, None,None,None]
            validity = prefix[:, :, None,None,None].to(cot.dtype).expand(-1,-1,1,cot.shape[-2],cot.shape[-1])
            x = encoder(torch.cat((cot, validity),2).transpose(1,2))
            x = x[torch.arange(len(lead),device=lead.device), :, 8+lead]
            return read(x.flatten(1))
        value = torch.cat((self.local_image(local_image), self.region_image(region_image),
            encode(local_cot,self.local_cot,self.local_read), encode(region_cot,self.region_cot,self.region_read),
            self.station(station), self.lead(lead)),1)
        return self.readout(value).squeeze(1)


class Rows(Dataset):
    def __init__(self, bank, base, split, arm):
        self.arm = arm
        folder = bank/split
        self.ids = np.load(folder/"head_row_ids.npy")
        seq = np.load(folder/"sequence_indices.npy")
        self.lookup = {int(v):i for i,v in enumerate(seq)}
        self.labels = {k:np.load(base/(k+".npy"),mmap_mode="r") for k in
                       ("sequence","station","lead","ghi","clear","weight","split")}
        if not np.all(self.labels["split"][self.ids] == (0 if split == "train" else 1)):
            raise RuntimeError("split key mismatch")
        self.arr = {k:np.load(folder/(k+".npy"),mmap_mode="r") for k in
            ("image_local","image_region","cot_local_history","cot_region_history",
             "cot_local_forecast","cot_region_forecast","cot_local_oracle","cot_region_oracle")}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        row = self.ids[item]
        i = self.lookup[int(self.labels["sequence"][row])]
        st, lead = int(self.labels["station"][row]), int(self.labels["lead"][row])
        kind = "oracle" if self.arm == "oracle" else "forecast"
        local = np.concatenate((self.arr["cot_local_history"][i,st], self.arr["cot_local_"+kind][i,st]),0)
        region = np.concatenate((self.arr["cot_region_history"][i], self.arr["cot_region_"+kind][i]),0)
        if self.arm == "zero":
            local.fill(0); region.fill(0)
        clear = float(self.labels["clear"][row])
        if clear <= 0:
            raise RuntimeError("nonpositive clear sky GHI in pilot cohort")
        return (np.array(self.arr["image_local"][i,st,lead]), np.array(self.arr["image_region"][i,lead]),
                local,region,st,lead,float(self.labels["ghi"][row])/clear,clear,
                float(self.labels["weight"][row]),int(row))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


@torch.no_grad()
def evaluate(model,loader):
    model.eval()
    rows,pred,truth,station,lead=[],[],[],[],[]
    for batch in loader:
        values=[x.cuda(non_blocking=True) for x in batch]
        local,region,lc,rc,st,ld,kt,clear,weight,rid=values
        z=model(local,region,lc,rc,st,ld).double()*clear
        rows.append(rid.cpu().numpy()); pred.append(z.cpu().numpy()); truth.append((kt*clear).cpu().numpy())
        station.append(st.cpu().numpy()); lead.append(ld.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in (("row_ids",rows),("pred_ghi",pred),("true_ghi",truth),("station",station),("lead",lead))}
    def metric(take):
        error=arrays["pred_ghi"][take]-arrays["true_ghi"][take]
        return {"n":len(error),"rmse":float(np.sqrt(np.mean(error**2))),"mae":float(np.mean(np.abs(error))),"bias":float(error.mean())}
    station_metrics={str(st):metric(arrays["station"]==st) for st in (0,1)}
    return {"station_equal_rmse":sum(v["rmse"] for v in station_metrics.values())/2,
            "stations":station_metrics,"leads":{str((ld+1)*15):metric(arrays["lead"]==ld) for ld in range(16)}},arrays


def main():
    parser=argparse.ArgumentParser()
    for key in ("bank","base","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--patience",type=int,default=5)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--minimum-updates",type=int,default=600)
    a=parser.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite pilot training")
    contract=json.loads((a.bank/"complete.json").read_text())
    assert contract["state"]=="COMPLETE_EXPLORATORY_REGIONAL_GHI_PILOT_BANK" and contract["test_used"] is False
    assert sha(a.base/"audit.json") == contract["base_pack_audit_sha256"]
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)
    profile=[]
    rows=Rows(a.bank,a.base,"train","forecast")
    total=torch.cuda.get_device_properties(0).total_memory
    chosen=None
    for size in (16,32,64,128,256,512,1024,2048):
        seed_all(a.seed)
        model=RegionalStationHead().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        try:
            batch=next(iter(DataLoader(rows,batch_size=size,num_workers=0)))
            values=[x.cuda() for x in batch]
            torch.cuda.reset_peak_memory_stats()
            start=time.monotonic()
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                z=model(*values[:6]); weight=values[8]
                loss=((z-values[6])**2*weight).sum()/weight.sum().clamp_min(1)
                loss.backward(); opt.step()
            torch.cuda.synchronize()
            rec={"batch":len(batch[0]),"peak_gib":torch.cuda.max_memory_allocated()/2**30,
                 "rows_per_second":3*len(batch[0])/(time.monotonic()-start)}
            profile.append(rec)
            if rec["peak_gib"]*2**30 < .75*total:
                chosen=rec["batch"]
        except torch.cuda.OutOfMemoryError:
            profile.append({"batch":size,"state":"OOM"})
            break
        finally:
            del model,opt
            if "values" in locals(): del values
            if "z" in locals(): del z
            if "loss" in locals(): del loss
            torch.cuda.empty_cache()
    save(a.output/"resource_profile.json",{"profile":profile,"selected_batch":chosen,
         "rule":"largest tested batch below75% VRAM; throughput recorded; identical batch across arms"})
    if chosen is None: raise RuntimeError("no safe batch")
    steps_per_epoch=math.ceil(len(rows)/chosen)
    epochs=max(a.epochs,math.ceil(a.minimum_updates/steps_per_epoch))
    patience=max(a.patience,math.ceil(120/steps_per_epoch))
    results={}
    for arm in ("zero","forecast","oracle"):
        seed_all(a.seed)
        model=RegionalStationHead().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        train=DataLoader(Rows(a.bank,a.base,"train",arm),batch_size=chosen,shuffle=True,
                         num_workers=a.workers,pin_memory=True)
        val=DataLoader(Rows(a.bank,a.base,"val",arm),batch_size=chosen,shuffle=False,
                       num_workers=a.workers,pin_memory=True)
        best=float("inf"); stale=0
        for epoch in range(epochs):
            model.train()
            for batch in train:
                values=[x.cuda(non_blocking=True) for x in batch]
                opt.zero_grad(set_to_none=True)
                z=model(*values[:6]); weight=values[8]
                loss=((z-values[6])**2*weight).sum()/weight.sum().clamp_min(1)
                if not torch.isfinite(loss): raise RuntimeError("nonfinite training loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True); opt.step()
            metrics,_=evaluate(model,val)
            if metrics["station_equal_rmse"] < best:
                best=metrics["station_equal_rmse"]; stale=0
                torch.save({"model":model.state_dict(),"epoch":epoch,"arm":arm},a.output/(arm+"_best.pt"))
            else: stale+=1
            rec={"state":"TRAINING_MATCHED_REGIONAL_GHI_PILOT","arm":arm,"epoch":epoch,
                 "val_rmse":metrics["station_equal_rmse"],"best_val_rmse":best,"test_used":False}
            save(a.output/"status.json",rec); print(json.dumps(rec),flush=True)
            if stale >= patience: break
        state=torch.load(a.output/(arm+"_best.pt"),weights_only=False)
        model.load_state_dict(state["model"])
        results[arm],arrays=evaluate(model,val)
        np.savez_compressed(a.output/(arm+"_validation_predictions.npz"),**arrays)
        save(a.output/(arm+"_validation_metrics.json"),results[arm])
    report={"state":"COMPLETE_MATCHED_REGIONAL_GHI_PILOT","test_used":False,"seed":a.seed,
        "batch":chosen,"epochs_max":epochs,"patience":patience,"results":results,
        "steps_per_epoch":steps_per_epoch,"maximum_optimizer_updates":steps_per_epoch*epochs,
        "loss":"same weighted MSE interval kt across arms; predicted GHI=kt*interval clear sky GHI",
        "scope":"single-seed exploratory pilot; oracle not deployable; CPP physical QA unresolved"}
    for arm in ("forecast","oracle"):
        report[arm+"_minus_zero_rmse"]=results[arm]["station_equal_rmse"]-results["zero"]["station_equal_rmse"]
    # Saved prediction keys and truth must match exactly before any claim about delta.
    saved={arm:np.load(a.output/(arm+"_validation_predictions.npz")) for arm in ("zero","forecast","oracle")}
    ref=saved["zero"]
    seqs=np.load(a.base/"sequence.npy",mmap_mode="r")[ref["row_ids"]]
    days=np.array([contract["initialization_BJT_days"]["val"][str(int(seq))] for seq in seqs])
    unique=np.unique(days)
    rng=np.random.default_rng(20260927)
    draws=rng.integers(0,len(unique),(2000,len(unique)))
    for arm in ("forecast","oracle"):
        other=saved[arm]
        for key in ("row_ids","true_ghi","station","lead"):
            assert np.array_equal(ref[key],other[key]), (arm,key)
        byday=np.zeros((len(unique),2,3),float)
        for di,day in enumerate(unique):
            for st in (0,1):
                mask=(days==day)&(ref["station"]==st)
                byday[di,st]=[mask.sum(),np.square(ref["pred_ghi"][mask]-ref["true_ghi"][mask]).sum(),
                             np.square(other["pred_ghi"][mask]-other["true_ghi"][mask]).sum()]
        sample=byday[draws].sum(1)
        keep=(sample[:,:,0]>0).all(1)
        delta=(np.sqrt(sample[keep,:,2]/sample[keep,:,0])-np.sqrt(sample[keep,:,1]/sample[keep,:,0])).mean(1)
        report[arm+"_date_bootstrap"]={"unit":"BJT initialization day","days":len(unique),"seed":20260927,
             "repetitions":2000,"valid_draws":int(keep.sum()),"station_equal_delta_rmse_95pct":np.quantile(delta,[.025,.975]).tolist()}
    save(a.output/"complete.json",report);save(a.output/"status.json",report)
    print(report["state"],flush=True)


if __name__ == "__main__":
    main()
