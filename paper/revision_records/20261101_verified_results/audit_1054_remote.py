"""Read only named train/val provenance; never open test/buffer assets."""
import json,hashlib
from pathlib import Path
import numpy as np
E=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
out={'scope':'named existing train/validation provenance only','test_payloads_read':0,'files':{}}
names=['R_pack_correctutc_attempt2/pack_status.json','R_correctutc_attempt3/run_metadata.json','cot_bank_correctutc_attempt3/BANK_REPORT.json','ghi_fourarm_pack_attempt5/pack_report.json']+[f'ghi_fourarm_seed{s}_attempt2/run_metadata.json' for s in [42,43,44]]
for n in names:
 p=E/n;out['files'][n]={'exists':p.exists()}
 if p.exists():out['files'][n].update(sha256=sha(p),data=json.loads(p.read_text()))
p=E/'R_correctutc_attempt3/best.pt'
out['R_checkpoint']={'path':str(p),'exists':p.exists()}
if p.exists():out['R_checkpoint']['sha256']=sha(p)
for n in ['R_pack_correctutc_attempt2/norm.json','R_pack_correctutc_attempt2/pack_status.json']:
 p=E/n
 if p.exists():out[n]=json.loads(p.read_text())
# Identity of actual geometry array; named old four-arm pack, not forecast/test.
p=E/'ghi_fourarm_pack_attempt5/geometry.npy'
out['ghi_geometry']={'path':str(p),'exists':p.exists()}
if p.exists():out['ghi_geometry'].update(sha256=sha(p),shape=list(np.load(p,mmap_mode='r').shape))
print(json.dumps(out))
