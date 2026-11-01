import json, hashlib
from pathlib import Path
import torch
root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
out={'checkpoints':[], 'receipts':[], 'test_buffer_payloads_read':False}
paths=['experiments/monthly_R_station16_20261013_v1/results/seed42_v1/best.pt',
       'experiments/r_solar_geometry_corrected_20261015/results/paired_v1/old/best.pt',
       'experiments/r_solar_geometry_corrected_20261015/results/paired_v1/corrected/best.pt',
       'runs/cot_repaired_seed42_v1/best.pt']
for rel in paths:
    p=root/rel; z=torch.load(p,map_location='cpu',weights_only=False)
    out['checkpoints'].append(dict(path=str(p),sha256=sha(p),keys=list(z),epoch=z.get('epoch'),metadata=z.get('metadata',{})))
for rel in ['experiments/r_solar_geometry_corrected_20261015/data/paired_common_v1/'+n for n in ['pack_audit.json','cot.npy','common_mask.npy','rows_supervised.csv']]+[
 'experiments/monthly_R_station16_20261013_v1/results/ghi_head_pack_new_r_v1/pack_manifest.json']:
    p=root/rel; r=dict(path=str(p),exists=p.is_file())
    if p.is_file():
        r['sha256']=sha(p)
        if p.suffix=='.json':r['json']=json.loads(p.read_text())
    out['receipts'].append(r)
print(json.dumps(out,indent=2,default=str))
