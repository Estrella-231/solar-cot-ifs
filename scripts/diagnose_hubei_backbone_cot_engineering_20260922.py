#!/usr/bin/env python3
"""Cross-region engineering diagnostic: invert frozen Hunan R on Hubei old-backbone outputs.

This is not a Hunan-paper experiment and does not validate COT accuracy.  It
only asks whether different extrapolators preserve a Hubei station-neighbourhood
cloud signal after a fixed, Hunan-trained retrieval transform.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

HUBEI_PATCH = (slice(103,119), slice(118,134))
AGRI_INDICES = np.asarray((0,1,2,3,4,5,8,9,10,11,12,13,14))

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--hubei-root",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--ddms-norm",type=Path,required=True)
    p.add_argument("--r-checkpoint",type=Path,required=True)
    p.add_argument("--r-code",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=56)
    return p.parse_args()

def rows(path):
    with path.open(newline="") as f:
        return {r["seq_id"]:r for r in csv.DictReader(f) if r["split"]=="val"}

def unit_to_native(x, cfg):
    c=cfg["channels"]
    lo=np.asarray([v["lower"] for v in c],np.float32)[None,:,None,None]
    hi=np.asarray([v["upper"] for v in c],np.float32)[None,:,None,None]
    inv=np.asarray([v["transform"]=="inverted_linear" for v in c])
    y=np.clip(x,0,1).copy()
    y[:,inv]=1-y[:,inv]
    return lo+y*(hi-lo)

def geometry(raw):
    soz=raw[18].astype(np.float32); soa=raw[17].astype(np.float32); saa=raw[15].astype(np.float32)
    cos_soz=np.cos(np.deg2rad(soz)); cos_raa=np.cos(np.deg2rad(soa-saa))
    day=(cos_soz>0).astype(np.float32)
    return np.stack([cos_soz,cos_raa,day],axis=0)

@torch.inference_mode()
def retrieve(model, native, geom, norm, device):
    # native [N,13,16,16]; geom [N,3,16,16]
    x=np.concatenate([native,geom],axis=1).astype(np.float32)
    x=(x-np.asarray(norm["mean"],np.float32)[None,:,None,None])/np.asarray(norm["std"],np.float32)[None,:,None,None]
    x[~np.isfinite(x)]=0
    out=[]
    for part in np.array_split(x, max(1, int(np.ceil(len(x)/256)))):
        out.append(model(torch.from_numpy(part).to(device)).float().cpu().numpy())
    return np.concatenate(out,axis=0)[:,0] # log1p COT

def main():
    a=parse(); a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.r_code))
    from train_cot_repaired import COTUNet
    ck=torch.load(a.r_checkpoint,map_location="cpu",weights_only=False)
    norm=ck["metadata"]["norm"]
    assert norm["feature_names"][:13]==["C01","C02","C03","C04","C05","C06","C09","C10","C11","C12","C13","C14","C15"]
    assert norm["feature_names"][13:]==["cosSOZ","cosRAA","day_mask"]
    model=COTUNet().eval()
    model.load_state_dict(ck["model"],strict=True)
    device=torch.device("cpu")
    cfg=json.loads(a.ddms_norm.read_text())
    lookup=rows(a.manifest)
    candidates=("motion_source","afno_temporal")
    ids=sorted(p.stem for p in (a.hubei_root/"evaluation/common56/motion_source/saved_samples").glob("*.npz"))[:a.limit]
    assert ids and all(i in lookup for i in ids)
    pred={m:[] for m in candidates}; truth=[]; metas=[]
    for sid in ids:
        paths=lookup[sid]["data_relpaths"].split("|")[8:24]
        g=np.stack([geometry(np.load(a.data_root/p)[...,HUBEI_PATCH[0],HUBEI_PATCH[1]]) for p in paths])
        # all output artifacts have the exact same truth and valid masks
        for mi,m in enumerate(candidates):
            z=np.load(a.hubei_root/"evaluation/common56"/m/"saved_samples"/f"{sid}.npz")
            native=unit_to_native(z["prediction"],cfg)[:,:,HUBEI_PATCH[0],HUBEI_PATCH[1]]
            pred[m].append(retrieve(model,native,g,norm,device))
            if mi==0:
                tn=unit_to_native(z["truth"],cfg)[:,:,HUBEI_PATCH[0],HUBEI_PATCH[1]]
                truth.append(retrieve(model,tn,g,norm,device))
        metas.append({"seq_id":sid,"bjt_start":str(np.load(a.hubei_root/"evaluation/common56/motion_source/saved_samples"/f"{sid}.npz")["bjt_start"])})
        print(json.dumps({"done":len(metas),"total":len(ids),"seq_id":sid}),flush=True)
    truth=np.asarray(truth,np.float32); pred={m:np.asarray(v,np.float32) for m,v in pred.items()}
    # select a real thick-cloud case by max over all leads/patch pixels
    flat=np.argmax(truth.reshape(len(ids),-1)); si=int(flat//truth[0].size)
    station=(8,8)
    payload={"scope":"Hubei cross-region engineering diagnostic only; not Hunan paper evidence and not COT-reference accuracy validation",
      "input":"Hubei saved common56 13-channel predictions inverse-DDMS-transformed to physical AGRI; exact per-target Hubei geometry from raw channel 15/17/18; fixed Hunan R",
      "models":list(candidates),"sample_count":len(ids),"canonical_leads_minutes":list(range(15,241,15)),
      "patch":"Hubei full crop rows 103:119, cols 118:134; station proxy (8,8)",
      "selected_visual_case":metas[si]}
    summary={}
    for m,x in pred.items():
       d=x-truth
       summary[m]={"map_mae_log1p":float(np.abs(d).mean()),"station_mae_log1p":float(np.abs(d[:,:,8,8]).mean()),
                   "local_peak_mae_log1p":float(np.abs(x.reshape(len(ids),16,-1).max(-1)-truth.reshape(len(ids),16,-1).max(-1)).mean())}
    payload["aggregate_vs_same_R_on_true_Hubei_AGRI"]=summary
    (a.output/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    np.savez_compressed(a.output/"retrievals.npz",truth=truth,**pred,metas=np.asarray(json.dumps(metas)))
    # physical picture uses expm1 but label is still R-retrieved COT proxy
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leads=[3,7,11,15]; names=["R(true Hubei AGRI)","R(Motion+Source)","R(AFNO+Transformer)"]
    arrays=[truth[si],pred["motion_source"][si],pred["afno_temporal"][si]]
    vmax=float(np.percentile(np.concatenate([q[leads].ravel() for q in arrays]),99))
    fig,axs=plt.subplots(3,4,figsize=(12,8),constrained_layout=True)
    for r,(name,q) in enumerate(zip(names,arrays)):
      for c,l in enumerate(leads):
        im=axs[r,c].imshow(np.expm1(q[l]),vmin=0,vmax=vmax,cmap="magma")
        axs[r,c].plot(8,8,"w*",ms=9,mec="k"); axs[r,c].set_xticks([]);axs[r,c].set_yticks([])
        axs[r,c].set_title(f"{name}\n+{(l+1)*15} min" if c==0 else f"+{(l+1)*15} min",fontsize=9)
    fig.colorbar(im,ax=axs,shrink=.78,label="R-retrieved COT proxy (exp(log1p)-1)")
    fig.suptitle(f"Hubei engineering stress test, {metas[si]['seq_id']} init {metas[si]['bjt_start']} BJT\nSame frozen Hunan R; not a cross-region COT validation",fontsize=12)
    fig.savefig(a.output/"selected_case_maps.png",dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    t=np.arange(1,17)*15
    for name,q in zip(names,arrays):
       axs[0].plot(t,np.expm1(q[:,8,8]),marker="o",label=name)
       axs[1].plot(t,np.expm1(q.reshape(16,-1).max(-1)),marker="o",label=name)
    axs[0].set(title="Station-proxy COT trajectory",xlabel="lead (min)",ylabel="R COT proxy")
    axs[1].set(title="Strongest cloud in 16x16 patch",xlabel="lead (min)",ylabel="R COT proxy")
    for ax in axs: ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Same selected Hubei case; retrieval domain is cross-region diagnostic only",fontsize=11)
    fig.savefig(a.output/"selected_case_trajectory.png",dpi=180)
    print(json.dumps({"status":"complete","output":str(a.output),"selected":metas[si],"summary":summary},indent=2))

if __name__=="__main__": main()