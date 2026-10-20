import ast,csv,json,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
root=Path(__file__).resolve().parent
source=(root/'legacy_solar_source.py').read_text(encoding='utf-8-sig')
functions='\n\n'.join(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('_julian_day','compute_solar_angles'))
ns={'np':np};exec(functions,ns);legacy=ns['compute_solar_angles']
fixed=functions.replace('eot = 4.0 * (','eot = 4.0 * (180.0 / np.pi) * (');assert fixed!=functions
nf={'np':np};exec(fixed,nf);correct=nf['compute_solar_angles']
st={'Niushou':(32.08,112.04),'Wangzhai':(32.04,112.12),'Mizhuang':(32.12,112.20)}
rows=list(csv.DictReader((root/'solar_geometry_checks_attempt1.csv').open()));cache={};a=[]
for r in rows:
 key=r['station'],r['target']
 if key not in cache:
  lat,lon=st[key[0]];bjt=datetime.strptime(key[1],'%Y%m%d%H%M');utc=bjt-timedelta(hours=8)
  z=legacy(lat,lon,utc)[0];zc=correct(lat,lon,utc)[0]
  cache[key]=(float(np.cos(np.deg2rad(z))),float(np.cos(np.deg2rad(zc))),float(z),float(zc))
 lc,cc,z,zc=cache[key];sc=float(r['stored_cosSOZ']);a.append([sc-lc,sc-cc,lc-cc,float(r['label_noaa_cosSOZ'])-cc,float(np.rad2deg(np.arccos(np.clip(sc,-1,1))))-zc])
a=np.array(a)
report={'samples':len(rows),'unique_station_times':len(cache),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'finding':'source eot treats radians as minutes; correction multiplies equation-of-time radian expression by 180/pi before factor4','legacy_source_cos_residual_max_abs':float(abs(a[:,0]).max()),'legacy_source_cos_residual_p99':float(np.quantile(abs(a[:,0]),.99)),'stored_vs_corrected_cos_max_abs':float(abs(a[:,1]).max()),'label_noaa_vs_corrected_cos_max_abs':float(abs(a[:,3]).max()),'stored_vs_corrected_zenith_max_degrees':float(abs(a[:,4]).max()),'stored_vs_corrected_zenith_p99_degrees':float(np.quantile(abs(a[:,4]),.99)),'decision':'preserve frozen S/R geometry for time-only paired experiment; source-formula parity replaces arbitrary .05 cosine threshold; known EOT error is not corrected by relabeling and must not be called high-precision solar geometry'}
print(json.dumps(report,indent=2));(root/'SOLAR_FORMULA_AUDIT.json').write_text(json.dumps(report,indent=2))
