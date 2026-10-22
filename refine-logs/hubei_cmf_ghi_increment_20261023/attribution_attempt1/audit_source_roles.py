import csv,json,numpy as np
from pathlib import Path
p=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_forecast_cot_increment_20261022/prepared_attempt1')
result={}
for st in ['Niushou','Wangzhai','Mizhuang']:
 rows=list(csv.DictReader((p/st/'rows.csv').open()));roles=np.load(p/st/'roles.npy');d={}
 for r,role in zip(rows,roles):
  if role not in ['C_fit','C_select']:continue
  key=role+':'+r['target_time_bjt'][:8];v=d.setdefault(key,{})
  f=r['source_role'];v[f]=v.get(f,0)+1
 result[st]=d
print(json.dumps(result,indent=2))
