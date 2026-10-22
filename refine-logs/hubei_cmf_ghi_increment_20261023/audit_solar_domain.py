"""Inspect only C_fit/C_select payloads before fitting; no mask modification."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
H=P.parent/'hubei_forecast_cot_increment_20261022/prepared_attempt1'
result={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/st; meta=json.loads((p/'meta.json').read_text())
    roles=np.load(p/'roles.npy'); take=np.flatnonzero(np.isin(roles,['C_fit','C_select']))
    rows=list(csv.DictReader((p/'rows.csv').open())); rr=[rows[i] for i in take]
    f=np.load(p/'future.npy',mmap_mode='r')[take]
    ff=np.load(p/'fusion_features.npy',mmap_mode='r')[take]
    fn=meta['future_cot_norm']; log=ff[:,279:288].astype('float64')*fn['log1p_std']+fn['log1p_mean']
    tau=np.expm1(np.maximum(log,0)); valid=ff[:,297:306].astype(bool)
    cosz=f[:,13].reshape(-1,9).astype('float64'); z=np.rad2deg(np.arccos(np.clip(cosz,-1,1)))
    outside=valid&(z>80.0001)
    def desc(mask):
        v=tau[mask]; return dict(n=int(mask.sum()),cot_quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist() if len(v) else [],sza_range=[float(z[mask].min()),float(z[mask].max())] if len(v) else [])
    r=dict(rows=len(take),valid=desc(valid),outside80=desc(outside),outside90=desc(valid&(z>90)),outside_positive_1e6=desc(outside&(tau>1e-6)),by_role={})
    for role in ['C_fit','C_select']:
        r['by_role'][role]=desc(outside&(roles[take]==role)[:,None])
    r['examples']=[dict(init=rr[i]['init_time_bjt'],target=rr[i]['target_time_bjt'],pixel=int(j),cot=float(tau[i,j]),sza=float(z[i,j])) for i,j in np.argwhere(outside)[:10]]
    r['meta_sha256']=hashlib.sha256((p/'meta.json').read_bytes()).hexdigest()
    result[st]=r
out=P/'SOLAR_DOMAIN_AUDIT_attempt1.json'
assert not out.exists()
out.write_text(json.dumps(dict(scope='C_fit/C_select only; no fitting; no official val/new-period/test labels',stations=result),indent=2))
print(json.dumps(result,indent=2))
