"""Fetch only current observed-input GHI validation target frames, without training."""
from pathlib import Path
import subprocess, io, json, hashlib
import numpy as np

P=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import csv,sys,io,json
import numpy as np
root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/hubei_same_time_geometry_20261102')
rows=list(csv.DictReader((root/'pack/rows.csv').open()))
wanted=sorted({r['time_bjt'] for r in rows if r['split']=='val'})
keys=list(csv.DictReader((root/'frames/keys.csv').open()))
lut={r['time_bjt']:i for i,r in enumerate(keys)}
ids=[lut[t] for t in wanted]
assert all(keys[i]['split']=='val' for i in ids)
a=np.load(root/'frames/raw.npy',mmap_mode='r')[ids]
c=np.load(root/'frames/cot_log1p.npy',mmap_mode='r')[ids]
assert a.shape[1:]==(16,16,16) and c.shape[1:]==(16,16)
buf=io.BytesIO()
np.savez_compressed(buf,times_bjt=np.array(wanted),raw=a,cot_log1p=c,
    frame_indices=np.array(ids),source=np.array([keys[i]['source'] for i in ids]))
sys.stdout.buffer.write(buf.getvalue())
'''
def main():
    cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','zjnu-hpc',
         '/public/home/slfu/miniconda3/envs/swc/bin/python','-']
    r=subprocess.run(cmd,input=REMOTE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
    if not r.stdout.startswith(b'PK'):
        raise RuntimeError(f'Invalid npz stream, return {r.returncode}, stderr {r.stderr.decode(errors="replace")[:1000]}')
    with np.load(io.BytesIO(r.stdout)) as z:
        audit={'frames':len(z['times_bjt']), 'dates':len(set(t[:8] for t in z['times_bjt'])),
            'shape':list(z['raw'].shape), 'test_payloads_read':0,'training_performed':False,
            'source_root':'/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/hubei_same_time_geometry_20261102',
            'ssh_exit_code':r.returncode,'sha256':hashlib.sha256(r.stdout).hexdigest()}
    (P/'validation_target_images.npz').write_bytes(r.stdout)
    (P/'FETCH.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))
if __name__=='__main__':main()
