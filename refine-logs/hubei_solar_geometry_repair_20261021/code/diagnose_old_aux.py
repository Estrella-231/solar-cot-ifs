import csv,numpy as np,json
from pathlib import Path
p=Path('/tmp/hubei_solar_audit_20261021_attempt2/metadata/validation_source_index.csv');r=list(csv.DictReader(p.open()));f={}
for q in r:
 for d,g in zip(q['data_paths'].split('|'),q['aux_paths'].split('|')):f[Path(d).stem]=(Path(d).name,Path(g).name)
s=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017')
for i,(u,(d,g)) in enumerate(sorted(f.items())):
 if i<9200:continue
 x=np.load(s/'data'/d,mmap_mode='r');z=np.load(s/'aux'/g);v=[float(np.max(abs(z['cosSOZ']-np.cos(np.deg2rad(x[18]))))),float(np.max(abs(z['cosRAA']-np.cos(np.deg2rad(x[17]-x[15]))))),int(np.count_nonzero(z['day_mask']!=x[19]))]
 if v[0]>=1e-4 or v[1]>=1e-4 or v[2]:print(json.dumps(dict(index=i,utc=u,diff=v,aux=g,raw_soz_range=[float(x[18].min()),float(x[18].max())])),flush=True)
