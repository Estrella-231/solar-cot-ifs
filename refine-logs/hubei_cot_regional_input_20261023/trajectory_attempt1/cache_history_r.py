"""Frozen regional-R training histories, sharded across allocated GPUs."""
import argparse,json,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path.insert(0,str(BASE))
from g4_inference import r_model,read_frame,sha,geometry
ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--world',type=int,required=True)
ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
s=json.loads((P/'SELECTION.json').read_text());audit=json.loads((P/'STAGE_AUDIT.json').read_text())
assert audit['state']=='PASS_TRAIN_HISTORY_LOCAL_HASHES' and audit['selection_sha256']==sha(P/'SELECTION.json')
rows=s['frames'][a.shard::a.world];assert all(r['split']=='train' for r in rows)
a.out.mkdir(exist_ok=False,parents=True);r,mu,sd=r_model();logs=[];masks=[];ir=[];keys=[];start=time.monotonic()
grid=dict(np.load(BASE/'grid_static256.npz'));max_geometry_difference=0.
ch=[0,1,2,3,4,5,8,9,10,11,12,13,14]
with torch.inference_mode():
    for j in range(0,len(rows),32):
        xx=[]
        for row in rows[j:j+32]:
            raw,g=read_frame(row,a.data)
            difference=float(np.max(abs(g-geometry(grid,datetime.strptime(row['bjt'],'%Y%m%d%H%M')))))
            max_geometry_difference=max(max_geometry_difference,difference);assert difference<1e-6,(row['bjt'],difference)
            v=np.concatenate([raw[ch],g])
            xx.append((v-mu)/sd);keys.append(row['bjt']);ir.append(raw[12])
            masks.append((g[2]>.5)&(g[0]>np.cos(np.deg2rad(80)))&(raw[8:15]>0).all(0))
        with torch.autocast('cuda',dtype=torch.float16):q=r(torch.from_numpy(np.stack(xx)).cuda()).float()
        assert torch.isfinite(q).all();logs.append(q.cpu().numpy()[:,0])
        print(json.dumps(dict(shard=a.shard,frames=min(j+32,len(rows)),total=len(rows))),flush=True)
np.savez(a.out/'history.npz',keys=np.asarray(keys),log_cot=np.concatenate(logs),valid=np.stack(masks),c13=np.stack(ir))
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='FROZEN_R_TRAIN_HISTORY_CACHED',frames=len(keys),
    seconds=time.monotonic()-start,sha256=sha(a.out/'history.npz'),R_frozen=True,test_buffer_payloads_read=0,
    outer_validation_payloads_read=0,GHI_labels_read=0,max_geometry_difference=max_geometry_difference),indent=2))
