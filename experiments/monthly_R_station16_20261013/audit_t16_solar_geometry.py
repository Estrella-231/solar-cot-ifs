#!/usr/bin/env python3
"""Small CPU SPA audit on frozen validation target times and the actual grid."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--deps',type=Path)
p.add_argument('--case-inputs',type=Path)
a=p.parse_args()
if a.deps:sys.path.insert(0,str(a.deps))
import pvlib
cases=json.loads(a.manifest.read_text(encoding='utf-8'))['selected_cases']
if a.case_inputs:
    with np.load(a.case_inputs,allow_pickle=False) as z:derived=z['geometry_cossoz_cosraa_daymask']
with np.load(a.data_root/'grid_static.npz',allow_pickle=False) as z:
    keys=z.files
    lat=np.asarray(z['lat'] if 'lat' in keys else z['latitude'])
    lon=np.asarray(z['lon'] if 'lon' in keys else z['longitude'])
    satellite_azimuth=np.asarray(z['saa'])
indices={'sili':(163,148),'zhujia':(73,150)}
records=[];seen=set();patches=[];patch_keys=[]
for case_i,case in enumerate(cases):
    st=case['station'];rr,cc=indices[st]
    for lead_i,text in enumerate(case['future_target_times_utc']):
        if (st,text) in seen:continue
        seen.add((st,text))
        utc=datetime.fromisoformat(text);bjt=utc+timedelta(hours=8)
        if a.case_inputs:
            cos,raa,day=map(float,derived[case_i,lead_i,:,8,8])
        else:
            files=list((a.data_root/'aux_256'/bjt.strftime('%Y%m')).glob(bjt.strftime('%Y%m%d%H%M')+'_*_aux.npz'))
            assert len(files)==1
            with np.load(files[0],allow_pickle=False) as z:
                cos=float(z['cosSOZ'][rr,cc]);raa=float(z['cosRAA'][rr,cc]);day=float(z['day_mask'][rr,cc])
        pos=pvlib.solarposition.get_solarposition(pd.DatetimeIndex([pd.Timestamp(text,tz='UTC')]),float(lat[rr,cc]),float(lon[rr,cc]),pressure=0)
        sza=float(pos['zenith'].iloc[0]);az=float(pos['azimuth'].iloc[0]);spa_cos=float(np.cos(np.deg2rad(sza)))
        r={'station':st,'target_time_utc':text,'grid_lat':float(lat[rr,cc]),'grid_lon':float(lon[rr,cc]),
           'input_cosSOZ':cos,'input_cosRAA':raa,'input_day_mask':day,'SPA_cosSOZ':spa_cos,
           'SPA_zenith_deg':sza,'SPA_solar_azimuth_deg':az,'cosSOZ_error':cos-spa_cos,
           'zenith_error_deg':float(np.rad2deg(np.arccos(np.clip(cos,-1,1))))-sza}
        sa=float(satellite_azimuth[rr,cc]);r['sat_azimuth_grid_degrees']=sa
        r['SPA_cosRAA']=float(np.cos(np.deg2rad(az-sa)))
        r['cosRAA_error']=raa-r['SPA_cosRAA']
        offsets=np.arange(-10,15.001,.25)
        fit_times=pd.DatetimeIndex([pd.Timestamp(text,tz='UTC')+pd.Timedelta(minutes=float(t)) for t in offsets])
        fit=pvlib.solarposition.get_solarposition(fit_times,float(lat[rr,cc]),float(lon[rr,cc]),pressure=0)
        objective=(np.cos(np.deg2rad(fit['zenith'].to_numpy()))-cos)**2+(np.cos(np.deg2rad(fit['azimuth'].to_numpy()-sa))-raa)**2
        r['best_two_cosine_clock_offset_min_diagnostic']=float(offsets[int(np.argmin(objective))])
        patch=np.empty((2,3,3),np.float32)
        for dr in range(-1,2):
            for dc in range(-1,2):
                local=pvlib.solarposition.get_solarposition(pd.DatetimeIndex([pd.Timestamp(text,tz='UTC')]),float(lat[rr+dr,cc+dc]),float(lon[rr+dr,cc+dc]),pressure=0)
                patch[0,dr+1,dc+1]=np.cos(np.deg2rad(float(local['zenith'].iloc[0])))
                patch[1,dr+1,dc+1]=np.cos(np.deg2rad(float(local['azimuth'].iloc[0])-float(satellite_azimuth[rr+dr,cc+dc])))
        patches.append(patch);patch_keys.append((st,text))
        records.append(r)
result={'state':'T16_SPA_CENTER_SOLAR_DIAGNOSTIC','pvlib_version':pvlib.__version__,'grid_fields':keys,
        'grid_static_sha256':hashlib.sha256((a.data_root/'grid_static.npz').read_bytes()).hexdigest(),
        'case_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        'case_inputs_sha256':hashlib.sha256(a.case_inputs.read_bytes()).hexdigest() if a.case_inputs else None,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'solar_method':'pvlib default nrel_numpy SPA; timezone UTC; pressure 0; geometric zenith',
        'n_unique_station_targets':len(records),'max_abs_cosSOZ_error':max(abs(r['cosSOZ_error']) for r in records),
        'max_abs_zenith_error_deg':max(abs(r['zenith_error_deg']) for r in records),
        'max_abs_cosRAA_error':max(abs(r['cosRAA_error']) for r in records),'records':records,'test_payload_opened':False}
a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
np.savez_compressed(a.output.with_suffix('.npz'),station=np.array([k[0] for k in patch_keys]),target_time_utc=np.array([k[1] for k in patch_keys]),geometry=np.stack(patches))
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
