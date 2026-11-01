import hashlib,json
from pathlib import Path
import numpy as np
base=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
root=base/'r_only_data_attempt6'
src=json.loads((base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_text())
assert src['state']=='PASS_SOURCE_R_ONLY_TRAIN_VAL' and src['count']==202
by={x['relative']:x for x in src['records']}; seen=[]
for j,(rel,rec) in enumerate(by.items()):
    p=root/rel
    if not p.is_file(): raise FileNotFoundError(str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    got=h.hexdigest()
    if got!=rec['sha256']: raise ValueError(('sha256 mismatch',rel,got,rec['sha256']))
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    if list(a.shape)!=rec['shape'] or str(a.dtype)!=rec['dtype'] or not np.isfinite(a).all():
        raise ValueError(('array mismatch',rel,a.shape,a.dtype))
    seen.append(rel)
    if j%32==0: print('VERIFIED',j+1,len(by),flush=True)
actual={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*.npy')}
assert actual==set(by)
report={'state':'PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256','count':len(seen),'source_audit_sha256':hashlib.sha256((base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_bytes()).hexdigest(),'all_hashes_match':True,'test_payloads_read':0,'buffer_payloads_read':0}
(base/'TRANSFER_AUDIT_attempt6.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
