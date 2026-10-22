"""Reuse the completed previous /dev/shm copy after count and hash witnesses."""
import argparse,csv,hashlib,json
from pathlib import Path
EXP=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022')
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
rows=list(csv.DictReader((EXP/'stage_attempt2/frame_manifest.csv').open()));assert len(rows)==2021
assert len(list((a.root/'data').glob('*.npy')))==2021
assert len(list((a.root/'aux').glob('*.npz')))==2021
checks=[]
for i in [0,len(rows)//2,len(rows)-1]:
    r=rows[i]
    for kind in ['data','aux']:
        p=a.root/kind/Path(r[kind+'_path']).name;h=hashlib.sha256(p.read_bytes()).hexdigest()
        assert h==r[kind+'_sha256'];checks.append(dict(path=str(p),sha256=h))
(EXP/'CACHE_REUSE_attempt2.json').write_text(json.dumps(dict(state='PASS',root=str(a.root),checks=checks,files_per_kind=2021),indent=2)+'\n')
print('LOCAL_CACHE_REUSE_PASS',a.root)
