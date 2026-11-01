#!/usr/bin/env python3
"""Extract frozen monthly-R dec1 features for keyed GHI train/validation rows.

This is an isolated T18 sidecar. It uses only the audited monthly train/val
candidate frames and GHI pack keys, and writes one [32,3,3] feature per
(station,target_time_utc). No labels or test payloads enter R inference.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

EXPECTED = {
    "candidate": "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807",
    "R": "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001",
    "R_source": "7a2e5cb441cdb5132a537cd03580017801595bdea839719d1773398d267210ce",
    "stations": ("sili", "zhujia"),
}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<22), b""): h.update(b)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    for name in ("candidate","data-root","monthly-norm","r-checkpoint","r-source","r-norm","cot-bank","ghi-pack","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--batch-frames",type=int,default=64)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(f"output exists: {a.output}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if sha(a.candidate)!=EXPECTED["candidate"]: raise ValueError("candidate manifest hash changed")
    if sha(a.r_checkpoint)!=EXPECTED["R"] or sha(a.r_source)!=EXPECTED["R_source"]:
        raise ValueError("R checkpoint/source does not match the audited monthly R")
    ghi_manifest=json.loads((a.ghi_pack/"pack_manifest.json").read_text())
    bank_manifest=json.loads((a.cot_bank/"bank_manifest.json").read_text())
    if ghi_manifest.get("state")!="COMPLETE" or ghi_manifest.get("test_payload_opened") is not False:
        raise ValueError("GHI pack is incomplete or test state is not closed")
    if bank_manifest.get("state")!="COMPLETE" or bank_manifest.get("R_checkpoint_sha256")!=EXPECTED["R"]:
        raise ValueError("R COT bank provenance/state failed")
    if bank_manifest.get("test_payload_opened") is not False: raise ValueError("test gate failed")

    # Restrict extraction to frames represented in the fixed GHI pack, and
    # verify every key against both independent indexes before touching AGRI.
    ghi_rows=read_csv(a.ghi_pack/"rows.csv")
    pack_keys={}
    for r in ghi_rows:
        key=(r["station"],r["target_time_utc"])
        if key in pack_keys: raise ValueError(f"duplicate GHI key: {key}")
        if r["split"] not in ("train","val"): raise ValueError("non train/val GHI row")
        pack_keys[key]=r
    candidates=read_csv(a.candidate)
    cand_by_time={}
    for r in candidates:
        if r["split"] not in ("train","val"): raise ValueError("non train/val candidate")
        t=r["target_time_utc"]
        if t not in cand_by_time: cand_by_time[t]={}
        if r["station"] in cand_by_time[t]: raise ValueError(f"duplicate candidate key {(r['station'],t)}")
        cand_by_time[t][r["station"]]=r
    bank_rows=read_csv(a.cot_bank/"frame_keys.csv")
    bank_by_time={r["target_time_utc"]:r for r in bank_rows}
    if len(bank_by_time)!=len(bank_rows): raise ValueError("duplicate bank timestamp")
    used_times=sorted({t for _,t in pack_keys})
    frame_rows=[]
    for t in used_times:
        if t not in cand_by_time or set(cand_by_time[t])!=set(EXPECTED["stations"]):
            raise ValueError(f"missing paired station candidate frame {t}")
        c0,c1=(cand_by_time[t][s] for s in EXPECTED["stations"])
        if c0["split"]!=c1["split"] or c0["agri_frame_relpath"]!=c1["agri_frame_relpath"]:
            raise ValueError(f"station candidate disagreement {t}")
        b=bank_by_time.get(t)
        if b is None or b["split"]!=c0["split"] or b["agri_frame_relpath"]!=c0["agri_frame_relpath"]:
            raise ValueError(f"COT bank key mismatch {t}")
        for s in EXPECTED["stations"]:
            bank_coord=tuple(map(int,b[f"{s}_grid_rc"].split(",")))
            candidate_coord=tuple(map(int,cand_by_time[t][s]["station_grid_rc"].split(",")))
            if bank_coord!=candidate_coord: raise ValueError(f"candidate/COT-bank station coordinate mismatch {(s,t)}")
            k=(s,t)
            if k in pack_keys:
                g=pack_keys[k]; c=cand_by_time[t][s]
                if g["split"]!=c["split"] or g["agri_frame_relpath"]!=c["agri_frame_relpath"]:
                    raise ValueError(f"GHI/candidate key mismatch {k}")
                if (int(g["row"]),int(g["col"]))!=candidate_coord:
                    raise ValueError(f"GHI/candidate station coordinate mismatch {k}")
        frame_rows.append({"split":c0["split"],"target_time_utc":t,"agri_frame_relpath":c0["agri_frame_relpath"],
                           "coords":[tuple(map(int,cand_by_time[t][s]["station_grid_rc"].split(","))) for s in EXPECTED["stations"]],
                           "bank_index":int(b["index"])})
    if len(frame_rows)!=int(ghi_manifest["unique_frames"]):
        raise ValueError(f"GHI unique frame coverage mismatch: {len(frame_rows)} != {ghi_manifest['unique_frames']}")

    rnorm=json.loads(a.r_norm.read_text()); rmean=np.asarray(rnorm["mean"],np.float32)[:,None,None]; rstd=np.asarray(rnorm["std"],np.float32)[:,None,None]
    snorm=json.loads(a.monthly_norm.read_text()); smean=np.asarray(snorm["mean"],np.float32)[:,None,None]; sstd=np.asarray(snorm["std"],np.float32)[:,None,None]
    if rmean.shape!=(16,1,1) or smean.shape!=(13,1,1) or not np.isfinite(rstd).all() or (rstd<=0).any():
        raise ValueError("normalization shape/value mismatch")
    if sha(a.monthly_norm)!=bank_manifest.get("monthly_norm_sha256") or sha(a.r_norm)!=bank_manifest.get("R_norm_sha256"):
        raise ValueError("normalization files differ from those used by the frozen COT bank")
    # The R source is imported unchanged. A forward hook captures the tensor
    # immediately after dec1 and before the COT head.
    sys.path.insert(0,str(a.r_source.parent))
    from train_station16_R import COTUNet
    ckpt=torch.load(a.r_checkpoint,map_location="cpu",weights_only=False)
    if ckpt.get("metadata",{}).get("training_script_sha256")!=sha(a.r_source): raise ValueError("checkpoint/source binding mismatch")
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    model=COTUNet().cuda().float().eval().requires_grad_(False)
    model.load_state_dict(ckpt["model"],strict=True); del ckpt
    captured=[]
    hook=model.dec1.register_forward_hook(lambda _m,_i,out: captured.append(out))
    a.output.mkdir(parents=True)
    keys_path=a.output/"frame_keys.csv"
    with keys_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["index","split","target_time_utc","agri_frame_relpath","sili_grid_rc","zhujia_grid_rc","cot_bank_index"])
        for i,fr in enumerate(frame_rows): w.writerow([i,fr["split"],fr["target_time_utc"],fr["agri_frame_relpath"],*[
            ",".join(map(str,rc)) for rc in fr["coords"]],fr["bank_index"]])
    features=np.lib.format.open_memmap(a.output/"dec1_station_3x3.npy",mode="w+",dtype=np.float32,shape=(len(frame_rows),2,32,3,3))
    support=np.lib.format.open_memmap(a.output/"input_valid_fraction.npy",mode="w+",dtype=np.float32,shape=(len(frame_rows),2))
    cot_bank=np.load(a.cot_bank/"cot_log1p.npy",mmap_mode="r",allow_pickle=False)
    errors=[]; pending=[]; pending_idx=[]; pending_sup=[]; n_done=0; max_cot_diff=0.
    def flush():
        nonlocal pending,pending_idx,pending_sup,n_done,max_cot_diff
        if not pending: return
        x=torch.from_numpy(np.stack(pending).reshape(-1,16,16,16)).cuda()
        captured.clear()
        with torch.inference_mode(),torch.autocast("cuda",enabled=False): pred=model(x).float()
        if len(captured)!=1 or captured[0].shape!=(len(pending)*2,32,16,16): raise RuntimeError("dec1 hook shape mismatch")
        feat=captured[0][:,:,7:10,7:10].reshape(len(pending),2,32,3,3).float().cpu().numpy()
        cot=pred.reshape(len(pending),2,1,16,16).cpu().numpy()
        if not np.isfinite(feat).all() or not np.isfinite(cot).all(): raise RuntimeError("nonfinite R feature/output")
        for j,idx in enumerate(pending_idx):
            features[idx]=feat[j]; support[idx]=pending_sup[j]
            diff=float(np.max(np.abs(cot[j]-cot_bank[frame_rows[idx]["bank_index"]])))
            max_cot_diff=max(max_cot_diff,diff)
        n_done+=len(pending_idx); features.flush();support.flush()
        (a.output/"progress.json").write_text(json.dumps({"state":"BUILDING","frames_complete":n_done,"frames_total":len(frame_rows),"test_payload_opened":False},indent=2)+"\n")
        print(f"PROGRESS {n_done}/{len(frame_rows)} cot_max_abs_diff={max_cot_diff:.8g}",flush=True)
        pending=[];pending_idx=[];pending_sup=[]

    # Reuse exact audited extraction/normalization logic from the bank builder.
    from build_monthly_new_R_cot_bank import get_station_inputs
    for i,fr in enumerate(frame_rows):
        try:
            xs,sv=get_station_inputs(a.data_root,fr,smean,sstd,rmean,rstd)
        except (OSError,ValueError,KeyError) as e:
            errors.append({"index":i,"target_time_utc":fr["target_time_utc"],"error":repr(e)}); continue
        pending.append(xs);pending_idx.append(i);pending_sup.append(sv)
        if len(pending)>=a.batch_frames: flush()
    flush();hook.remove()
    if errors: raise RuntimeError(f"input source errors {len(errors)}; refusing incomplete bank")
    if max_cot_diff>0.002: raise RuntimeError(f"R output replay differs from frozen bank: {max_cot_diff}")

    # Train-only per-channel scaling, computed from the keyed GHI train rows.
    frame_index={r["target_time_utc"]:i for i,r in enumerate(frame_rows)}
    train_feats=[];val_keys=[]
    for (station,t),r in pack_keys.items():
        if r["split"]=="train": train_feats.append(features[frame_index[t],EXPECTED["stations"].index(station)])
        else: val_keys.append([station,t])
    if len(train_feats)!=int(ghi_manifest["train"]): raise RuntimeError("train feature support count mismatch")
    z=np.asarray(train_feats,dtype=np.float64).transpose(1,0,2,3)
    mean=z.mean(axis=(1,2,3)); std=np.maximum(z.std(axis=(1,2,3)),1e-6)
    np.savez(a.output/"train_feature_norm.npz",mean=mean.astype(np.float64),std=std.astype(np.float64),fit_rows=np.asarray([len(train_feats)],np.int64))
    manifest={"state":"COMPLETE","split_id":ghi_manifest["split_id"],"sample_key":"(station,target_time_utc)",
              "GHI_pack_rows":len(pack_keys),"GHI_train_rows":len(train_feats),"GHI_val_rows":len(val_keys),"unique_frames":len(frame_rows),
              "feature":"R dec1 activation immediately before COT head, station crop [7:10,7:10]","feature_shape":"[32,3,3] per GHI key",
              "R_checkpoint_sha256":sha(a.r_checkpoint),"R_source_sha256":sha(a.r_source),"candidate_sha256":sha(a.candidate),
              "GHI_pack_manifest_sha256":sha(a.ghi_pack/"pack_manifest.json"),"COT_bank_manifest_sha256":sha(a.cot_bank/"bank_manifest.json"),
              "max_abs_COT_replay_difference_vs_saved_bank":max_cot_diff,"train_only_feature_norm":{"mean":mean.tolist(),"std":std.tolist(),"fit_rows":len(train_feats)},
              "array_sha256":{"dec1_station_3x3.npy":sha(a.output/"dec1_station_3x3.npy"),"frame_keys.csv":sha(keys_path),"train_feature_norm.npz":sha(a.output/"train_feature_norm.npz")},
              "test_payload_opened":False,"source_errors":errors}
    (a.output/"feature_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="train_only_feature_norm"},indent=2),flush=True)

if __name__=="__main__": main()
