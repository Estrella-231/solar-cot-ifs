import csv, numpy as np
from pathlib import Path
base=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
rp=base.parent/'hubei_cloud_probability_20261020/R_pack_correctutc_attempt2'
rows=list(csv.DictReader((rp/'rows.csv').open(encoding='utf-8-sig')))
cp=np.load(base.parent/'hubei_cloud_probability_20261020/cpp_attempt2/cpp_supervision_trainval.npz',allow_pickle=False)
rkeys=[r['time_bjt'] for r in rows]; ckeys=[str(x) for x in cp['time_bjt']]
print('R rows',len(rkeys),'unique',len(set(rkeys)),'first',rkeys[:12])
print('CPP keys',len(ckeys),'unique',len(set(ckeys)),'dtype',cp['time_bjt'].dtype,'first',ckeys[:12])
print('same ordered',rkeys==ckeys,'same sets',set(rkeys)==set(ckeys),'r-only',len(set(rkeys)-set(ckeys)),'cpp-only',len(set(ckeys)-set(rkeys)))
if rkeys!=ckeys:
  for i,(a,b) in enumerate(zip(rkeys,ckeys)):
    if a!=b:
      print('first order mismatch',i,repr(a),repr(b));break
for t in rkeys[:5]: print('key',repr(t),'split R',rows[rkeys.index(t)]['split'],'split CPP',str(cp['split'][ckeys.index(t)]))
