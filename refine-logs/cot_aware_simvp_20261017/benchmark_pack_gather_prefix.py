"""Read-only microbenchmark/equivalence check on a live completed pack prefix."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from build_future_ghi_head_pack import context_source

def rows(path, limit):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for _, r in zip(range(limit), csv.DictReader(f))]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--exp",type=Path,required=True)
    p.add_argument("--limit",type=int,default=1024)
    a=p.parse_args(); exp=a.exp
    out=exp/"results/future_ghi_oof_head_pack_v1/train"
    bank=exp/"results/future_simvp_r_bank_oof_v1"
    ks=rows(out/"keys.csv",a.limit)
    cross=np.asarray([int(r["crosswalk_row_index"]) for r in ks],np.int64)
    with np.load(exp/"results/future_ghi_crosswalk_keys.npz",allow_pickle=False) as z:
        sc=z["split_code"][cross]; sq=z["sequence_index_within_split"][cross]
        st=z["station_index"][cross]; li=z["lead_index"][cross]
    lookup={}
    with open(bank/"index.csv",newline="",encoding="utf-8") as f:
        for i,r in enumerate(csv.DictReader(f)):
            if r["split"]=="train": lookup[int(r["sequence_index_within_split"])]=i
    bi=np.asarray([lookup[int(x)] for x in sq],np.int64)
    arr={n:np.load(bank/n,mmap_mode="r",allow_pickle=False) for n in (
        "history_agri_raw.npy","history_geometry.npy","history_cot_log1p.npy",
        "forecast_agri_raw.npy","forecast_geometry.npy","forecast_cot_log1p.npy")}
    dst={n:np.load(out/n,mmap_mode="r",allow_pickle=False)[:len(cross)] for n in (
        "agri_raw.npy","geometry.npy","cot_log1p.npy")}
    specs=(("agri_raw.npy","history_agri_raw.npy","forecast_agri_raw.npy"),
           ("geometry.npy","history_geometry.npy","forecast_geometry.npy"),
           ("cot_log1p.npy","history_cot_log1p.npy","forecast_cot_log1p.npy"))
    t0=time.perf_counter()
    ref={n:np.empty_like(dst[n]) for n,_,_ in specs}
    for i in range(len(cross)):
        for name,hn,fn in specs:
            for pos,off in enumerate((-2,-1,0)):
                hist,src=context_source(li[i:i+1],off); hist=bool(hist[0]); src=int(src[0])
                ref[name][i,pos]=arr[hn if hist else fn][bi[i],int(st[i]),src]
    row_seconds=time.perf_counter()-t0
    t1=time.perf_counter()
    vec={n:np.empty_like(dst[n]) for n,_,_ in specs}
    for name,hn,fn in specs:
        channels=vec[name].shape[2]
        for pos,off in enumerate((-2,-1,0)):
            hist,src=context_source(li,off)
            block=np.empty((len(cross),channels,3,3),np.float32)
            h=np.flatnonzero(hist); f=np.flatnonzero(~hist)
            if len(h): block[h]=arr[hn][bi[h],st[h],src[h]]
            if len(f): block[f]=arr[fn][bi[f],st[f],src[f]]
            vec[name][:,pos]=block
    vec_seconds=time.perf_counter()-t1
    out_check={n:bool(np.array_equal(ref[n],dst[n]) and np.array_equal(vec[n],dst[n])) for n,_,_ in specs}
    print(json.dumps({"rows":len(cross),"all_match_existing_original_pack":all(out_check.values()),
        "per_array_match":out_check,"rowwise_seconds":row_seconds,"batch_seconds":vec_seconds,
        "speedup_on_gather_only":row_seconds/max(vec_seconds,1e-9),
        "rowwise_rows_per_second":len(cross)/row_seconds,
        "batch_rows_per_second":len(cross)/vec_seconds},indent=2))
    if not all(out_check.values()): raise SystemExit("ACTUAL_PREFIX_GATHER_MISMATCH")

if __name__=="__main__": main()
