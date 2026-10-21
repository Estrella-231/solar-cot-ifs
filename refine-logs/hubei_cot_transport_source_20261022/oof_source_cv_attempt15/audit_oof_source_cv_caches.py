#!/usr/bin/env python3
"""Independent audit of paired OOF/full-R train-date-holdout caches."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from train_station_patch_cot_forecaster import ForecastWindowDataset, make_inner_split, support_summary

AUDIT_STATE = "PASS_STATION_PATCH_CACHE_AND_PAIR_AUDIT_TEST_BUFFER_CLOSED"
ARRAYS = ("geometry", "history_cot_log", "history_valid", "cpp_cot", "cpp_valid")
LEADS = tuple(range(15, 241, 15))
COS80 = math.cos(math.radians(80.0))
LOG101 = math.log(101.0)

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def read_csv(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--base-cache",type=Path,required=True)
    ap.add_argument("--oof-bank",type=Path,required=True)
    ap.add_argument("--oof-rows",type=Path,required=True)
    args=ap.parse_args()
    root=args.root
    build=json.loads((root/"BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    if build.get("state")!="COMPLETE_OOF_SOURCE_PAIRED_CACHE_BUILD": raise ValueError("bad build state")
    for k in ("test_payloads_read","buffer_payloads_read","official_monthly_validation_payloads_read"):
        if build.get(k)!=0: raise ValueError(f"{k} is nonzero")
    base_rows=read_csv(args.base_cache/"bank_rows.csv")
    base_train={r["time_bjt"]:int(r["index"]) for r in base_rows if r["split"].lower()=="train"}
    base_val={r["time_bjt"] for r in base_rows if r["split"].lower()=="val"}
    base_arrays={n:np.load(args.base_cache/f"{n}.npy",mmap_mode="r",allow_pickle=False) for n in ARRAYS}
    z=np.load(args.oof_bank,allow_pickle=False)
    oof_times=[str(x) for x in z["time_bjt"].tolist()]
    oof_i={t:i for i,t in enumerate(oof_times)}
    oof_rows=read_csv(args.oof_rows)
    if len(oof_i)!=len(oof_times) or set(oof_i)!=set(base_train): raise ValueError("OOF bank key set differs from train")
    if [r["time_bjt"] for r in oof_rows]!=oof_times: raise ValueError("OOF bank row order mismatch")
    oof_cot=z["cot"]; oof_day=z["day_mask"]
    out={}
    arm_state={}
    for arm in ("fullr_control","oof_train_source"):
        cache=root/arm/"cache"
        report=json.loads((cache/"cache_report.json").read_text(encoding="utf-8"))
        if report.get("state")!="COMPLETE_HUBEI_TRAIN_ONLY_OUTER_DATE_HOLDOUT_CACHE": raise ValueError("bad cache state")
        part=report["partition"]
        if any(part.get(k)!=0 for k in ("test_payloads_read","buffer_payloads_read","original_monthly_validation_payloads_read")):
            raise ValueError(f"payload gate failed for {arm}")
        rows=read_csv(cache/"bank_rows.csv")
        times=[r["time_bjt"] for r in rows]
        if len(times)!=len(set(times)) or set(times)&base_val or not set(times)<=set(base_train):
            raise ValueError(f"row key/split leakage for {arm}")
        arrays={n:np.load(cache/f"{n}.npy",mmap_mode="r",allow_pickle=False) for n in ARRAYS}
        n=len(rows)
        exp={"geometry":(n,3,16,16),"history_cot_log":(n,1,16,16),"history_valid":(n,1,16,16),
             "cpp_cot":(n,1,16,16),"cpp_valid":(n,1,16,16)}
        for name,a in arrays.items():
            if a.shape!=exp[name]: raise ValueError(f"shape mismatch {arm}/{name}: {a.shape}")
        train_i=[]; val_i=[]; old_i=[]
        for i,r in enumerate(rows):
            t=r["time_bjt"]; oi=int(r["source_train_index"])
            if base_train.get(t)!=oi: raise ValueError(f"source index mismatch {arm}/{t}")
            old_i.append(oi)
            (train_i if r["split"]=="train" else val_i).append(i)
            for name in ("geometry","cpp_cot","cpp_valid"):
                if not np.array_equal(arrays[name][i],base_arrays[name][oi]):
                    raise ValueError(f"base {name} changed for {arm}/{t}")
            if arm=="fullr_control" or r["split"]=="val":
                for name in ("history_cot_log","history_valid"):
                    if not np.array_equal(arrays[name][i],base_arrays[name][oi]):
                        raise ValueError(f"full-R {name} mismatch for {arm}/{t}")
            else:
                j=oof_i[t]; raw=np.asarray(oof_cot[j,0],dtype=np.float32)
                geom=np.asarray(arrays["geometry"][i],dtype=np.float32)
                day=np.asarray(oof_day[j,0],dtype=bool)
                if not np.array_equal(day,geom[2]>.5): raise ValueError(f"OOF day mismatch {t}")
                valid=day & (geom[2]>.5) & (geom[0]>COS80) & np.isfinite(raw) & (raw>=0) & (raw<=100)
                safe=np.where(valid,raw,0).astype(np.float32)
                expected=(np.log1p(safe)/np.float32(LOG101)).astype(np.float32)
                if not np.array_equal(arrays["history_valid"][i,0],valid.astype(np.uint8)):
                    raise ValueError(f"OOF valid mask mismatch {t}")
                if not np.array_equal(arrays["history_cot_log"][i,0],expected):
                    raise ValueError(f"OOF transformed map mismatch {t}")
        pairs=read_csv(cache/"forecast_pairs.csv")
        seen=set()
        for r in pairs:
            key=(r["split"],r["init_time_bjt"],int(r["lead_minutes"]))
            if key in seen: raise ValueError(f"duplicate pair {arm}/{key}")
            seen.add(key)
            ii,ti=int(r["init_index"]),int(r["target_index"])
            if rows[ii]["time_bjt"]!=r["init_time_bjt"] or rows[ti]["time_bjt"]!=r["target_time_bjt"]:
                raise ValueError(f"pair index mismatch {arm}/{key}")
            if rows[ii]["split"]!=r["split"] or rows[ti]["split"]!=r["split"]:
                raise ValueError(f"cross split pair {arm}/{key}")
            if int(np.asarray(arrays["cpp_valid"][ti,0],dtype=bool).sum())!=int(r["target_valid_pixels"]):
                raise ValueError(f"support count mismatch {arm}/{key}")
        ds_train=ForecastWindowDataset(cache,"train")
        fit_dates,inner_dates,_,inner_split=make_inner_split(cache)
        inner=ForecastWindowDataset(cache,"train",init_dates=inner_dates,allowed_target_dates=inner_dates)
        outer=ForecastWindowDataset(cache,"val")
        ins=support_summary(inner); outs=support_summary(outer)
        if len(ds_train)<100 or len(inner)<100 or ins["minimum_pairs_per_lead"]<100 or ins["minimum_target_dates_per_lead"]<12:
            raise ValueError(f"inner gate failed {arm}: {ins}")
        if len(outer)<100 or outs["minimum_pairs_per_lead"]<200 or outs["minimum_target_dates_per_lead"]<12:
            raise ValueError(f"outer gate failed {arm}: {outs}")
        out[arm]={"cache_sha256":{n:sha256(cache/f"{n}.npy") for n in ARRAYS},
                  "bank_rows_sha256":sha256(cache/"bank_rows.csv"),"forecast_pairs_sha256":sha256(cache/"forecast_pairs.csv"),
                  "train_windows":len(ds_train),"inner_holdout_windows":len(inner),"outer_holdout_windows":len(outer),
                  "inner_holdout_dates":inner_split["inner_holdout_dates"],"inner_support":ins,"outer_support":outs}
        arm_state[arm]=(cache,rows,arrays,seen)
    a,b=arm_state["fullr_control"],arm_state["oof_train_source"]
    if out["fullr_control"]["bank_rows_sha256"]!=out["oof_train_source"]["bank_rows_sha256"]: raise ValueError("arm row manifests differ")
    if out["fullr_control"]["forecast_pairs_sha256"]!=out["oof_train_source"]["forecast_pairs_sha256"]: raise ValueError("arm pair manifests differ")
    if a[3]!=b[3]: raise ValueError("paired arm pairs differ")
    if a[3] != b[3]: raise ValueError("paired pair-key sets differ")
    if out["fullr_control"]["cache_sha256"]["geometry"]!=out["oof_train_source"]["cache_sha256"]["geometry"]: raise ValueError("geometry differs")
    for name in ("cpp_cot","cpp_valid"):
        if out["fullr_control"]["cache_sha256"][name]!=out["oof_train_source"]["cache_sha256"][name]: raise ValueError(f"target differs: {name}")
    val_idx=[i for i,r in enumerate(a[1]) if r["split"]=="val"]
    if not np.array_equal(a[2]["history_cot_log"][val_idx],b[2]["history_cot_log"][val_idx]): raise ValueError("outer val history COT differs")
    if not np.array_equal(a[2]["history_valid"][val_idx],b[2]["history_valid"][val_idx]): raise ValueError("outer val history mask differs")
    receipt={"state":AUDIT_STATE,"scope":"original Hubei train dates only; internal outer date-block holdout",
             "official_monthly_validation_payloads_read":0,"test_payloads_read":0,"buffer_payloads_read":0,
             "outer_holdout_dates":build["outer_holdout_dates"],"purged_dates":build["purged_dates"],
             "paired_contract":"same keys, geometry, CPP targets/masks and outer histories; train-history R source is the only changed factor",
             "arms":out}
    for arm in arm_state:
        audit_dir=root/arm/"cache_audit"; audit_dir.mkdir(parents=True,exist_ok=True)
        (audit_dir/"cache_audit.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    (root/"CACHE_AUDIT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=="__main__": main()
