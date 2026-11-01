"""Seed-42 controls for the spatial-COT pilot.

AC_shuffle destroys spatial order while preserving the per-sample COT values.
A_proxy replaces the map by its spatial mean, preserving amplitude only.
Both controls use the same frozen base pack, cohort, head and GHI loss as B1.
"""
import argparse, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from spatial_head_smoke_v1 import SpatialHead

ROOT_ROWS, TRAIN_ROWS, VAL_ROWS = 637902, 538053, 99849
GROUPS = ("AC_shuffle", "A_proxy")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024**2), b""): h.update(b)
    return h.hexdigest()
def atomic(path, value):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)
def save(path, value):
    tmp = path.with_suffix(".tmp"); torch.save(value, tmp); tmp.replace(path)
def seed(v):
    np.random.seed(v); torch.manual_seed(v); torch.cuda.manual_seed_all(v)
def ghi_loss(pred, truth, clear, weight):
    return ((((pred.float()*clear.float()-truth.float())/1000.)**2)*weight.float()).mean()
def make_cot(cot, mode):
    if mode == "A_proxy": return cot.mean(dim=(-1,-2), keepdim=True).expand_as(cot)
    if mode == "AC_shuffle":
        out = cot.clone(); flat = out.flatten(1); idx = torch.arange(256, device=cot.device)
        center = 8*16+8; others = torch.cat((idx[:center], idx[center+1:]))
        perm = torch.roll(others, shifts=37)
        flat[:, others] = flat[:, perm]
        return flat.view_as(cot)
    raise ValueError(mode)
def predict(model, data, ids, mode):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(data["x"][ids], make_cot(data["cot_map"][ids], mode),
                     torch.zeros((len(ids), 4), device="cuda"), data["g"][ids],
                     data["station"][ids], data["lead"][ids]).float()
def evaluate(model, data, ids, mode, batch, keep=False):
    model.eval(); sse=torch.zeros(2,dtype=torch.float64,device="cuda"); counts=torch.zeros(2,dtype=torch.int64,device="cuda"); out=[]
    with torch.no_grad():
        for ix in ids.split(batch):
            pred=predict(model,data,ix,mode); residual=pred.double()*data["clear"][ix]-data["ghi"][ix]
            for s in (0,1):
                m=data["station"][ix]==s; sse[s]+=residual[m].square().sum(); counts[s]+=m.sum()
            if keep: out.append(pred.cpu().numpy())
    rmse=torch.sqrt(sse/counts); result={"station_equal_ghi_rmse":float(rmse.mean()),"station_ghi_rmse":rmse.cpu().tolist(),"station_counts":counts.cpu().tolist()}
    return result, np.concatenate(out) if keep else None
def main(root,out):
    root,out=root.resolve(),out.resolve()
    if out.exists(): raise RuntimeError("refuse to overwrite control directory")
    base=root/"data/head_pack_trainval_20260908_v1"; side=root/"data/head_spatial_cot_trainval_20260917_v2"
    ba=json.loads((base/"audit.json").read_text()); ma=json.loads((side/"audit.json").read_text())
    if ba["test_used"] or ma["test_used"] or ma["state"]!="COMPLETE_SPATIAL_COT_SIDECAR": raise RuntimeError("input contract")
    if sha(base/"audit.json")!=ma["base_pack_audit_sha256"] or sha(side/"cot_map.npy")!=ma["arrays"]["cot_map.npy"]["sha256"]: raise RuntimeError("hash binding")
    for spec in ba["arrays"].values():
        if sha(base/spec["name"])!=spec["sha256"]: raise RuntimeError("base hash drift")
    names=("x","g","ghi","clear","station","lead","split","weight")
    data={n:torch.from_numpy(np.load(base/(n+".npy"),allow_pickle=False)).cuda() for n in names}
    data["cot_map"]=torch.from_numpy(np.load(side/"cot_map.npy",allow_pickle=False)).cuda()
    train=torch.where(data["split"]==0)[0]; val=torch.where(data["split"]==1)[0]
    if len(train)!=TRAIN_ROWS or len(val)!=VAL_ROWS: raise RuntimeError("split")
    out.mkdir(parents=True)
    contract={"state":"EXPLORATORY_B2_CONTROLS_CPP_PROVENANCE_UNRESOLVED","test_used":False,"groups":list(GROUPS),"seed":42,"epochs":10,"batch":2048,"lr":3e-4,"loss":"mean(w*((pred_kt*clear-ghi)/1000)^2)","base_pack_audit_sha256":sha(base/"audit.json"),"sidecar_audit_sha256":sha(side/"audit.json"),"script_sha256":sha(__file__)}
    atomic(out/"execution_contract.json",contract); report={"state":"RUNNING_EXPLORATORY_B2_CONTROLS",**contract,"runs":{}}
    for mode in GROUPS:
        seed(42); model=SpatialHead().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); best=float("inf"); best_epoch=None; epochs=[]
        for ep in range(10):
            model.train(); gen=torch.Generator(device="cuda").manual_seed(42+ep); order=train[torch.randperm(len(train),generator=gen,device="cuda")]; total=0.; start=time.monotonic()
            for ix in order.split(2048):
                opt.zero_grad(set_to_none=True); pred=predict(model,data,ix,mode); loss=ghi_loss(pred,data["ghi"][ix],data["clear"][ix],data["weight"][ix]); loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
            met,_=evaluate(model,data,val,mode,2048); rec={"epoch":ep,"train_weighted_normalized_ghi_mse":total/len(train),**met,"seconds":time.monotonic()-start}; atomic(out/(mode+"_epoch_%02d.json"%ep),rec); print(mode,json.dumps(rec),flush=True); epochs.append(rec)
            if met["station_equal_ghi_rmse"]<best: best=met["station_equal_ghi_rmse"]; best_epoch=ep; save(out/(mode+"_best.pt"),{"model":model.state_dict(),"mode":mode,"epoch":ep,"contract":contract})
        ck=torch.load(out/(mode+"_best.pt"),map_location="cpu"); model.load_state_dict(ck["model"],strict=True); model.cuda().eval(); met,pred=evaluate(model,data,val,mode,2048,True); pp=out/(mode+"_validation_predictions.npz"); np.savez(pp,pack_row=val.cpu().numpy(),pred_kt=pred,observed_ghi=data["ghi"][val].cpu().numpy(),clear_sky_ghi=data["clear"][val].cpu().numpy(),station=data["station"][val].cpu().numpy(),lead=data["lead"][val].cpu().numpy()); report["runs"][mode]={"best_epoch_0based":best_epoch,**met,"epochs":epochs,"checkpoint_sha256":sha(out/(mode+"_best.pt")),"predictions_sha256":sha(pp)}; atomic(out/"progress.json",report); del model,opt; gc.collect(); torch.cuda.empty_cache()
    report["state"]="COMPLETE_EXPLORATORY_B2_CONTROLS"; report["next"]="No seeds, IFS, or test until controls are reviewed."; atomic(out/"complete.json",report); print("SPATIAL_B2_CONTROLS_COMPLETE",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--allow-unverified-cpp",action="store_true"); a=p.parse_args()
    if not a.allow_unverified_cpp: raise RuntimeError("requires explicit authorization")
    main(a.root,a.out)
