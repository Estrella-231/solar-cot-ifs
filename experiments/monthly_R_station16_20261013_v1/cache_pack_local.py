#!/usr/bin/env python3
import hashlib, json, shutil, socket, sys
from pathlib import Path
source, target, output = map(Path, sys.argv[1:4])
names = ('x_raw.npy', 'cot.npy', 'mask.npy', 'rows.csv', 'source_coverage.csv', 'pack_status.json')
required = sum((source / n).stat().st_size for n in names)
if shutil.disk_usage(target).free < required + 2 * 1024**3:
    raise RuntimeError('Insufficient node-local disk space for package staging')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
files={}
for name in names:
    shutil.copy2(source/name,target/name)
    sh,th=sha(source/name),sha(target/name)
    if sh!=th: raise RuntimeError(f'Local cache copy differs: {name}')
    files[name]={'bytes':(target/name).stat().st_size,'sha256':th}
r={'host':socket.gethostname(),'source':str(source),'local_pack':str(target),'files':files}
output.write_text(json.dumps(r,indent=2)+'\n')
print('MONTHLY_R_LOCAL_CACHE_READY',json.dumps(r),flush=True)
