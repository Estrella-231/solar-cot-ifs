import sys,json,csv
from pathlib import Path
import numpy as np,torch
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/public/home/slfu/ttzhou/swc/irradiance_forecast")
HB=ROOT/"HubeiBackboneBench_20260905"; TC=ROOT/"TemporalCorrDiff"; OUT=HB/"evaluation/cot_crossregion_engineering_quick5_20260922"
sys.path.insert(0,str(ROOT/"SolarCOTIFS_20260907/scripts"))
from train_cot_repaired import COTUNet
def geo(raw):
 s=np.cos(np.deg2rad(raw[18])); return np.stack([s,np.cos(np.deg2rad(raw[17]-raw[15])),(s>0).astype(np.float32)]).astype(np.float32)
def rget(model,a,g,norm):
 x=np.concatenate([a,g],1); x=(x-np.array(norm["mean"],np.float32)[None,:,None,None])/np.array(norm["std"],np.float32)[None,:,None,None]; x[~np.isfinite(x)]=0
 with torch.no_grad():return model(torch.from_numpy(x)).numpy()[:,0]
def getpaths(sid,split):
 with open(TC/"manifests/agri_hubei_simvp_8to16.csv") as f:
  row=next(r for r in csv.DictReader(f) if r["seq_id"]==sid and r["split"]==split)
 return row["data_relpaths"].split("|")[8:24]
ck=torch.load(ROOT/"SolarCOTIFS_20260907/runs/cot_repaired_seed42_v1/best.pt",map_location="cpu",weights_only=False);m=COTUNet().eval();m.load_state_dict(ck["model"]);norm=ck["metadata"]["norm"]; patch=(slice(103,119),slice(118,134))
# SimVP same val case; prior quick bundle has true, motion and AFNO
z=np.load(OUT/"retrievals.npz",allow_pickle=True); metas=json.loads(str(z["metas"])); i=next(i for i,q in enumerate(metas) if q["seq_id"]=="val_000036")
sim=np.load("/tmp/simvp_val_000036_native.npz")["prediction"][:,:,patch[0],patch[1]]
g=np.stack([geo(np.load("/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed/data"/Path(p))[...,patch[0],patch[1]]) for p in getpaths("val_000036","val")])
simr=rget(m,sim,g,norm); arr=[z["truth"][i],z["motion_source"][i],z["afno_temporal"][i],simr]; names=["R(true Hubei AGRI)","R(Motion+Source)","R(AFNO+Transformer)","R(SimVP)"]; leads=[3,7,11,15]; vmax=np.percentile(np.concatenate([x[leads].ravel() for x in arr]),99)
fig,ax=plt.subplots(4,4,figsize=(12,10),constrained_layout=True)
for r,(n,x) in enumerate(zip(names,arr)):
 for c,l in enumerate(leads):
  im=ax[r,c].imshow(np.expm1(x[l]),cmap="magma",vmin=0,vmax=vmax);ax[r,c].plot(8,8,"w*",mec="k");ax[r,c].set_xticks([]);ax[r,c].set_yticks([]);ax[r,c].set_title((n+"\n" if c==0 else "")+f"+{(l+1)*15} min",fontsize=9)
fig.colorbar(im,ax=ax,label="R-retrieved COT proxy");fig.suptitle("Hubei cross-region engineering diagnostic, val_000036; same frozen Hunan R",fontsize=12);fig.savefig(OUT/"selected_case_maps_with_simvp.png",dpi=180);plt.close(fig)
# Earthformer: raw saved z score, fixed train sample only
e=np.load(HB/"evaluation/earthformer_official_overfit32_v1/fixed_train_median.npz",allow_pickle=True); stats=json.loads((TC/"manifests/agri_zscore_train_all_v1.json").read_text())["agri"]
mean=np.asarray(stats["mean"],np.float32)[None,:,None,None];std=np.asarray(stats["std"],np.float32)[None,:,None,None]
ep=e["prediction"]*std+mean; et=e["truth"]*std+mean; sid=str(e["seq_id"]); gg=np.stack([geo(np.load("/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed/data"/Path(p))[...,patch[0],patch[1]]) for p in getpaths(sid,"train")])
er=rget(m,ep[:,:,patch[0],patch[1]],gg,norm);etr=rget(m,et[:,:,patch[0],patch[1]],gg,norm); vmax=np.percentile(np.r_[etr[leads].ravel(),er[leads].ravel()],99)
fig,ax=plt.subplots(2,4,figsize=(12,5.4),constrained_layout=True)
for r,(n,x) in enumerate(zip(["R(true Hubei AGRI)","R(Earthformer overfit32)"],[etr,er])):
 for c,l in enumerate(leads):
  im=ax[r,c].imshow(np.expm1(x[l]),cmap="magma",vmin=0,vmax=vmax);ax[r,c].plot(8,8,"w*",mec="k");ax[r,c].set_xticks([]);ax[r,c].set_yticks([]);ax[r,c].set_title((n+"\n" if c==0 else "")+f"+{(l+1)*15} min",fontsize=9)
fig.colorbar(im,ax=ax,label="R-retrieved COT proxy");fig.suptitle(f"Earthformer fixed training overfit check, {sid}: not a validation comparison",fontsize=12);fig.savefig(OUT/"earthformer_fixed_train_cot_maps.png",dpi=180);plt.close(fig)
print(json.dumps({"simvp":"ok","earthformer_seq":sid,"earthformer_rmse_by_lead":e["rmse_by_lead"].tolist()}))