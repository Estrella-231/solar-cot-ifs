#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
BASE=Path("/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021 ".strip())
PREP=Path('/dev/shm/hubei_solar_audit_20261021_attempt10')
MAN=BASE/'prepared_attempt10/stage_manifest.csv'
RPACK=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2')
GRID=BASE/'grid_static256.npz'
sys.path.insert(0,str(BASE/'code_pipeline_attempt10'))
from solar_geometry_noaa import compute_solar_angles
with MAN.open(encoding='utf-8-sig',newline='') as f: srows=list(csv.DictReader(f))
with (RPACK/'rows.csv').open(encoding='utf-8-sig',newline='') as f: rrows=list(csv.DictReader(f))
assert len(srows)==10473 and len(rrows)==3479
sl=(slice(103,119),slice(118,134)); g=dict(np.load(GRID,allow_pickle=False)); lat,lon,saa=(g[k][sl] for k in ('lat','lon','saa'))
s_lut={r['utc']:r for r in srows}; checked=0; first=None; maxes={'day':0.0,'cosSOZ':0.0,'cosRAA':0.0,'SOA_channel':0.0,'SOZ_channel':0.0,'day_channel':0.0,'prep_aux':0.0}
for rr in rrows:
 b=rr['time_bjt']; dt=datetime.strptime(b,'%Y%m%d%H%M'); u=(dt-timedelta(hours=8)).strftime('%Y%m%d%H%M')
 if u not in s_lut: continue
 sr=s_lut[u]; ap=Path(sr['aux_path']); dp=Path(sr['data_path'])
 if not ap.is_file() or not dp.is_file(): raise FileNotFoundError(f'{u}: data={dp.is_file()} aux={ap.is_file()}')
 z,az,day=compute_solar_angles(lat,lon,dt-timedelta(hours=8)); az=np.where(day>.5,az,0.)
 exp=np.stack((np.cos(np.deg2rad(z)),np.cos(np.deg2rad(az-saa)),day)).astype(np.float32)
 aux=np.load(ap,allow_pickle=False); got=np.stack([aux[k] for k in ('cosSOZ','cosRAA','day_mask')])
 raw=np.load(dp,mmap_mode='r',allow_pickle=False)
 errs={'day':float(np.abs(exp[2]-got[2]).max()),'cosSOZ':float(np.abs(exp[0]-got[0]).max()),'cosRAA':float(np.abs(exp[1]-got[1]).max()),'SOA_channel':float(np.abs(np.where(day>.5,az,0.)-raw[17,sl[0],sl[1]]).max()),'SOZ_channel':float(np.abs(z-raw[18,sl[0],sl[1]]).max()),'day_channel':float(np.abs(day-raw[19,sl[0],sl[1]]).max())}
 prep_aux=PREP/'aux'/ap.name
 if prep_aux.is_file():
  pa=np.load(prep_aux,allow_pickle=False);errs['prep_aux']=max(float(np.abs(got[k]-pa[k]).max()) for k in ('cosSOZ','cosRAA','day_mask'))
 for k,v in errs.items(): maxes[k]=max(maxes[k],v)
 checked+=1
 if first is None and (errs['day']>0 or errs['cosSOZ']>=1e-6 or errs['cosRAA']>=1e-6):
  first={'R_time_bjt':b,'UTC':u,'S_aux_path':str(ap),'S_data_path':str(dp),'S_aux_key':sr['utc'],'S_bjt':sr['bjt'],'errors':errs,'geometry_at_center':{'expected':exp[:,8,6].tolist(),'cached':got[:,8,6].tolist()},'prep_aux_exists':prep_aux.is_file()}
  break
result={'state':'DIAG_COMPLETE' if first else 'NO_MISMATCH_IN_FIRST_OVERLAPS','checked_overlaps':checked,'first_mismatch':first,'max_abs_errors':maxes,'test_payloads_read':0,'buffer_payloads_read':0}
out=BASE/'GEOMETRY_AUX_MISMATCH_DIAG_attempt11.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
