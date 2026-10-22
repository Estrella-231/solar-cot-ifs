"""Check actual contextual values, not only supervised pixels, before fp16."""
import argparse,csv,json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--roles',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
rows=list(csv.DictReader((a.pack/'rows.csv').open()));roles={r['time_bjt']:r['role'] for r in json.loads(a.roles.read_text())['roles']}
x=np.load(a.pack/'x_norm.npy',mmap_mode='r');mask=np.load(a.pack/'mask.npy',mmap_mode='r')
norm=json.loads((a.pack/'norm.json').read_text());groups={};bad=[]
for i,row in enumerate(rows):
    role=roles[row['time_bjt']];supported=bool(mask[i].any())
    group=groups.setdefault(role,dict(frames=0,supported_frames=0,above_fp16_range_frames=0,supported_above_fp16_range_frames=0,
                                     max_abs_input=0.,max_abs_day_input=0.))
    maximum=float(abs(x[i]).max());daymax=float(abs(x[i,15]).max());exceeds=maximum>65504
    assert np.isfinite(x[i]).all()
    group['frames']+=1;group['supported_frames']+=supported;group['above_fp16_range_frames']+=exceeds
    group['supported_above_fp16_range_frames']+=supported and exceeds
    group['max_abs_input']=max(group['max_abs_input'],maximum);group['max_abs_day_input']=max(group['max_abs_day_input'],daymax)
    if exceeds:bad.append(dict(time_bjt=row['time_bjt'],role=role,supported=supported,max_abs=maximum,max_abs_day=daymax))
identity=norm['mean'][15]==0 and norm['std'][15]==1
if identity:assert np.all((x[:,15]==0)|(x[:,15]==1))
passed=identity and not bad
r=dict(state='PASS_NORMALIZED_CONTEXT_FP16_RANGE' if passed else 'FAIL_BINARY_STANDARDIZATION_OR_INPUT_RANGE',
       day_mask_mean=norm['mean'][15],day_mask_std=norm['std'][15],day_mask_identity=identity,
       groups=groups,bad_frames=bad,pack=str(a.pack),training_started=False,test_payloads_read=0)
a.out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='bad_frames'}),flush=True)
