"""Historical R256 metadata-only time/provenance audit; no image arrays."""
import json, hashlib
from pathlib import Path
from collections import Counter
root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
out={'root':str(root),'receipts':[],'metadata_rows':0,'included':Counter(),'time_mismatches':0,'test_buffer_payloads_read':False}
for p in sorted(root.glob('r256*/run_metadata.json')):
    r={'path':str(p),'metadata':json.loads(p.read_text()),'sha256':sha(p)}
    best=p.parent/'best.pt'
    if best.is_file():r['checkpoint_sha256']=sha(best)
    out['receipts'].append(r)
cache=root/'agri_cpp_pair_bank_v1'; p=cache/'index.json'
if p.is_file():
    index=json.loads(p.read_text());out['cache_index']={'sha256':sha(p),'json':index}
    for s in index['shards']:
        p=cache/s['directory']/'records.json'
        for r in json.loads(p.read_text()):
            out['metadata_rows']+=1
            if r['utc']!=Path(r['agri_relpath']).stem:out['time_mismatches']+=1
            if r['included_for_R']:out['included'][r['split']]+=1
for p in [root/'cpp_bank_v1/index.json',root/'regional_cpp_bank_v1/index.json']:
    if p.is_file():out.setdefault('cpp_indices',[]).append({'path':str(p),'sha256':sha(p),'json':json.loads(p.read_text())})
print(json.dumps(out,indent=2))
