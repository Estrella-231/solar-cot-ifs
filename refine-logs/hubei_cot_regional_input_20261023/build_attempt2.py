"""Rebuild only the new regional pack with binary day-mask identity."""
import json,shutil,subprocess,sys
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
old=Path('/dev/shm/hubei_regional_r_pack_20261023_attempt1')
out=Path('/dev/shm/hubei_regional_r_pack_20261023_attempt2')
assert not out.exists() and shutil.disk_usage('/dev/shm').free>20_000_000_000
subprocess.run([sys.executable,str(P/'build_regional_pack.py'),'--manifest',str(P/'selected_cpp_agri.csv'),
                '--cpp-ready',str(P/'source_ready_attempt1'),'--out',str(out)],check=True)
subprocess.run([sys.executable,str(P/'audit_normalized_context.py'),'--pack',str(out),
                '--roles',str(P/'R_INTERNAL_ROLES.json'),'--out',str(out/'NORMALIZATION_AUDIT.json')],check=True)
audit=json.loads((out/'NORMALIZATION_AUDIT.json').read_text());assert audit['state']=='PASS_NORMALIZED_CONTEXT_FP16_RANGE'
before=json.loads((old/'SHA256.json').read_text());after=json.loads((out/'SHA256.json').read_text())
for name in ('rows.csv','target_log1p.npy','mask.npy','support.json'):assert before[name]==after[name],name
x=np.load(old/'x_norm.npy',mmap_mode='r');y=np.load(out/'x_norm.npy',mmap_mode='r')
assert x.shape==y.shape
for i in range(len(x)):assert np.array_equal(x[i,:15],y[i,:15]),i
result=dict(state='PASS_BINARY_ONLY_PACK_REPAIR',pack=str(out),frames=len(x),
            unchanged=['row keys','physical targets','supervision masks','first15 normalized channels','date roles'],
            changed='only day_mask transform: physical 0/1 identity instead of daylight mean1/std1e-6',
            old_pack_preserved=str(old),training_started=False,test_payloads_read=0)
(P/'G2_ATTEMPT2_COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
