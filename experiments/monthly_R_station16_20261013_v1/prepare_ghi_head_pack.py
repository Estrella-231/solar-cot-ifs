#!/usr/bin/env python3
"""Join the frozen train/val GHI rows to observed AGRI, geometry, and new-R COT.

This is a train/validation-only materializer. It keys every modality by station
and canonical UTC target time, never by row position, and rejects test rows.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("sili", "zhujia")
EXPECTED_R_SHA = "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001"
EXPECTED_CANDIDATE_SHA = "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807"

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<22), b""): h.update(b)
    return h.hexdigest()

def aux_path(root: Path, rel: str) -> Path:
    t=datetime.strptime(Path(rel).stem,"%Y%m%d%H%M")+timedelta(hours=8)
    matches=sorted((root/"aux_256"/t.strftime("%Y%m")).glob(t.strftime("%Y%m%d%H%M")+"_*_aux.npz"))
    if len(matches)!=1: raise FileNotFoundError(f"geometry sidecar count={len(matches)} for {t}")
    return matches[0]

def main():
    p=argparse.ArgumentParser()
    for k in ("labels","candidate","data_root","cot_bank","output"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if sha256(a.candidate)!=EXPECTED_CANDIDATE_SHA: raise ValueError("candidate manifest hash changed")
    manifest=json.loads((a.cot_bank/"bank_manifest.json").read_text())
    if manifest.get("R_checkpoint_sha256")!=EXPECTED_R_SHA or manifest.get("test_payload_opened") is not False:
        raise ValueError("COT bank provenance/test gate failed")
    if manifest.get("state") not in ("COMPLETE","COMPLETE_WITH_UNAVAILABLE_INPUT_FRAMES"):
        raise ValueError("COT bank is not complete")
    bank_keys={}
    with (a.cot_bank/"frame_keys.csv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key=r["target_time_utc"]
            if key in bank_keys: raise ValueError("duplicate COT bank timestamp")
            bank_keys[key]=(int(r["index"]),r["split"],r["agri_frame_relpath"],r["sili_grid_rc"],r["zhujia_grid_rc"])
    cot=np.load(a.cot_bank/"cot_log1p.npy",mmap_mode="r",allow_pickle=False)
    support=np.load(a.cot_bank/"input_valid_fraction.npy",mmap_mode="r",allow_pickle=False)
    if cot.shape!=(len(bank_keys),2,1,16,16) or support.shape!=(len(bank_keys),2): raise ValueError("bank arrays differ from index")
    rows=[]; seen=set()
    with a.labels.open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("new_split") not in ("train","val"): raise ValueError("non-train/val label row encountered")
            if r.get("main_mask","").lower()!="true": continue
            if r["station"] not in STATIONS: continue
            if not np.isfinite(float(r["observed_ghi_15min_wm2"])) or float(r["clear_sky_ghi_15min_wm2"])<=0: continue
            key=r["target_time_utc"]
            sample_key=(r["station"],key)
            if sample_key in seen: raise ValueError(f"duplicate GHI sample key {sample_key}")
            seen.add(sample_key)
            if key not in bank_keys: continue
            idx,split,rel,sili_rc,zhujia_rc=bank_keys[key]
            if split!=r["new_split"] or rel!=r["agri_frame_relpath"]: raise ValueError("label/COT-bank key mismatch")
            si=STATIONS.index(r["station"]); rc=tuple(map(int,(sili_rc if si==0 else zhujia_rc).split(",")))
            cot_patch=np.asarray(cot[idx,si,0,7:10,7:10],dtype=np.float32)
            if cot_patch.shape!=(3,3) or not np.isfinite(cot_patch).all(): continue
            rows.append({"split":split,"station":r["station"],"target_time_utc":key,"agri_frame_relpath":rel,"row":rc[0],"col":rc[1],"bank_index":idx,"station_index":si,"ghi":float(r["observed_ghi_15min_wm2"]),"clear":float(r["clear_sky_ghi_15min_wm2"]),"kt":float(r["kt"]),"input_valid_fraction":float(support[idx,si]),"cot_log1p":cot_patch})
    if not rows: raise RuntimeError("no common labeled/COT-bank samples")
    n=len(rows); a.output.mkdir(parents=True)
    ag=np.lib.format.open_memmap(a.output/"agri_raw.npy",mode="w+",dtype=np.float32,shape=(n,13,3,3))
    av=np.lib.format.open_memmap(a.output/"agri_valid.npy",mode="w+",dtype=np.uint8,shape=(n,13,3,3))
    geo=np.lib.format.open_memmap(a.output/"geometry.npy",mode="w+",dtype=np.float32,shape=(n,2,3,3))
    co=np.lib.format.open_memmap(a.output/"cot_log1p.npy",mode="w+",dtype=np.float32,shape=(n,1,3,3))
    root=a.data_root
    frame_groups={}
    for i,r in enumerate(rows): frame_groups.setdefault(r["agri_frame_relpath"],[]).append((i,r))
    packed=0;next_report=2000
    for rel,group in frame_groups.items():
        raw=np.load(root/"data"/rel,mmap_mode="r",allow_pickle=False)
        if raw.shape!=(20,256,256) or raw.dtype!=np.float32: raise ValueError(f"bad AGRI source shape/dtype: {rel} {raw.shape} {raw.dtype}")
        with np.load(aux_path(root,rel),allow_pickle=False) as z:
            bits_all=z["agri_valid_bits"]
            for i,r in group:
                r0,c0=r["row"]-1,r["col"]-1
                if r0<0 or c0<0 or r0+3>256 or c0+3>256: raise ValueError(f"3x3 crop out of bounds for {rel}")
                bits=np.asarray(bits_all[r0:r0+3,c0:c0+3])
                patch=np.asarray(raw[np.asarray(CHANNELS),r0:r0+3,c0:c0+3],dtype=np.float32)
                mask=np.stack([((bits&(1<<k))!=0) for k in CHANNELS]) & np.isfinite(patch)
                ag[i]=np.where(mask,patch,0); av[i]=mask
                r["agri_valid_fraction_3x3"]=float(mask.mean())
                geo_patch=np.stack([z[k][r0:r0+3,c0:c0+3] for k in ("cosSOZ","cosRAA")]).astype(np.float32)
                if not np.isfinite(geo_patch).all(): raise ValueError(f"nonfinite solar geometry for {rel}")
                geo[i]=geo_patch
                co[i,0]=r["cot_log1p"];packed+=1
        if packed>=next_report:
            ag.flush(); av.flush(); geo.flush(); co.flush(); print(f"PACK {packed}/{n}",flush=True)
            next_report+=2000
    ag.flush();av.flush();geo.flush();co.flush()
    with (a.output/"rows.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!="cot_log1p"]);w.writeheader()
        for r in rows: w.writerow({k:v for k,v in r.items() if k!="cot_log1p"})
    split=np.asarray([r["split"] for r in rows]); valid=np.asarray(av,dtype=bool)
    x=np.asarray(ag); train=split=="train"
    means=[];stds=[]
    for c in range(13):
        v=x[train,c][valid[train,c]]
        means.append(float(v.mean()));stds.append(float(max(v.std(),1e-6)))
    cot_train=np.asarray(co[train],dtype=np.float64)
    cmean=float(cot_train.mean());cstd=float(max(cot_train.std(),1e-6))
    meta={"state":"COMPLETE","rows":n,"unique_frames":len(frame_groups),"train":int(train.sum()),"val":int((~train).sum()),"stations":{s:int(sum(r["station"]==s for r in rows)) for s in STATIONS},"split_id":"hunan_monthly_block_2d_buffer1d_seed42_v1","sample_key":"(station,target_time_utc)","label":"15-minute mean GHI and kt from frozen trainval label join; main_mask only","AGRI_channels":list(CHANNELS),"AGRI_norm_mean":means,"AGRI_norm_std":stds,"COT_transform":"log1p from frozen new-monthly-R output; train-sample mean/std","COT_norm_mean":cmean,"COT_norm_std":cstd,"geometry":"physical cosSOZ/cosRAA, no spatial pooling","test_payload_opened":False,"R_checkpoint_sha256":EXPECTED_R_SHA,"candidate_sha256":sha256(a.candidate),"label_sha256":sha256(a.labels),"bank_manifest":manifest,"bank_indices_hash":sha256(a.cot_bank/"frame_keys.csv"),"support_summary":{"mean_R_input_valid_fraction_by_station":[float(np.mean([r["input_valid_fraction"] for r in rows if r["station"]==s])) for s in STATIONS],"mean_AGRI_valid_fraction_3x3_by_station":[float(np.mean([r["agri_valid_fraction_3x3"] for r in rows if r["station"]==s])) for s in STATIONS]},"array_sha256":{"agri_raw.npy":sha256(a.output/"agri_raw.npy"),"agri_valid.npy":sha256(a.output/"agri_valid.npy"),"geometry.npy":sha256(a.output/"geometry.npy"),"cot_log1p.npy":sha256(a.output/"cot_log1p.npy"),"rows.csv":sha256(a.output/"rows.csv")}}
    (a.output/"pack_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({k:meta[k] for k in ("state","rows","train","val","stations","support_summary","test_payload_opened")},indent=2))
if __name__=="__main__": main()
