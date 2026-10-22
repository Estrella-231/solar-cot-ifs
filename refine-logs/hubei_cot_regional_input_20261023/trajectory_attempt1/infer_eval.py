"""Frozen per-model val inference; inputs contain no future cloud/CPP labels."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from trajectory_model import TrajectorySource
from train_trajectory import sha
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
selection=json.loads((P/'fits_attempt1/ARM_SELECTION.json').read_text());assert selection['state']=='SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL'
spec=next(r for r in selection['runs'] if Path(r['run'])==a.run);ckpath=a.run/'best.pt';assert sha(ckpath)==spec['best_sha256']
meta=json.loads((a.pack/'REPORT.json').read_text());assert meta['state']=='FROZEN_OUTER_VAL_INPUTS_READY_AFTER_ARM_SELECTION'
assert meta['arm_selection_sha256']==sha(P/'fits_attempt1/ARM_SELECTION.json')
ck=torch.load(ckpath,map_location='cpu');assert ck['binding']['model_source_sha256']==sha(P/'trajectory_model.py')
assert ck['binding']['arm']==spec['arm'] and ck['binding']['seed']==spec['seed']
model=TrajectorySource().cuda().eval();model.load_state_dict(ck['model'],strict=True)
rows=list(csv.DictReader((a.pack/'pairs.csv').open()));assert all(r['role']=='outer_val' for r in rows)
log=np.load(a.pack/'history_log.npy',mmap_mode='r');valid=np.load(a.pack/'history_valid.npy',mmap_mode='r')
geo=np.load(a.pack/'target_geometry.npy',mmap_mode='r');out=[];started=time.monotonic()
with torch.inference_mode():
    for start in range(0,len(rows),32):
        rr=rows[start:start+32];hi=np.array([[int(v) for v in r['history_indices'].split('|')] for r in rr]);ti=np.array([int(r['target_index']) for r in rr])
        values=[np.asarray(log[hi]),np.asarray(valid[hi]),np.array([[float(r['motion_dy']),float(r['motion_dx'])] for r in rr],np.float32),
                np.array([int(r['lead']) for r in rr],np.float32),np.asarray(geo[ti])]
        args=[torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in values]
        with torch.autocast('cuda',dtype=torch.float16):pred=model(*args,arm=spec['arm']).float()
        assert torch.isfinite(pred).all()
        small=torch.stack([pred[:,0,r-1:r+2,c-1:c+2] for r,c in [(111,124),(112,126),(110,128)]],1)
        out.append(small.cpu().numpy())
assert not a.out.exists();a.out.mkdir(parents=True)
np.savez(a.out/'predictions.npz',init=np.array([r['init'] for r in rows]),target=np.array([r['target'] for r in rows]),
         lead=np.array([int(r['lead']) for r in rows]),pred_log=np.concatenate(out))
report=dict(state='FROZEN_TRAJECTORY_VAL_PREDICTIONS_SAVED',arm=spec['arm'],seed=spec['seed'],rows=len(rows),
    checkpoint_sha256=sha(ckpath),prediction_sha256=sha(a.out/'predictions.npz'),seconds=time.monotonic()-started,
    CPP_or_GHI_labels_used_as_inputs=False,test_buffer_payloads_read=0)
(a.out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
