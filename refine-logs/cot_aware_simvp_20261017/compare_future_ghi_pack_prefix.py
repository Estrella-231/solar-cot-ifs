"""Compare optimized benchmark prefix against the live original builder output."""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument("--original",type=Path,required=True); p.add_argument("--candidate",type=Path,required=True); p.add_argument("--rows",type=int,required=True); a=p.parse_args()
arrays=("agri_raw.npy","geometry.npy","cot_log1p.npy","ghi.npy","clear.npy")
checks={}
for name in arrays:
    x=np.load(a.original/name,mmap_mode="r",allow_pickle=False)[:a.rows]
    y=np.load(a.candidate/name,mmap_mode="r",allow_pickle=False)
    checks[name]=bool(np.array_equal(x,y))
with open(a.original/"keys.csv","rb") as f: old=b"".join(next(f) for _ in range(a.rows+1))
new=(a.candidate/"keys.csv").read_bytes()
checks["keys.csv"]=old==new
print(json.dumps({"rows":a.rows,"exact_prefix_match":all(checks.values()),"checks":checks},indent=2))
if not all(checks.values()): raise SystemExit("PACK_PREFIX_MISMATCH")
