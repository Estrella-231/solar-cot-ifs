"""Seven historical CPP frames, conditional geometric scenarios, no GHI scoring."""
import csv, hashlib, importlib.util, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
R=P.parents[1]
GRID=R/'refine-logs/hubei_history_cot_confirmation_20261022/grid_static256.npz'
SOURCE=P/'height_extract_attempt1'
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
ray=load_module('ray',R/'experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/project_cth_shadows.py')
solar=load_module('solar',R/'refine-logs/hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/solar_geometry_noaa.py')
with np.load(GRID,allow_pickle=False) as f: lat=f['lat'].astype(float);lon=f['lon'].astype(float)
assert lat.shape==lon.shape==(256,256)
centers={'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}
manifest=json.loads((SOURCE/'COMPLETE.json').read_text())
records=[]
for frame in manifest['records']:
    file=SOURCE/(frame['time_bjt']+'.npz')
    assert hashlib.sha256(file.read_bytes()).hexdigest()==frame['output_sha256']
    assert frame['metadata']['variable_attributes']['CTH']['Unit']=='km'
    with np.load(file,allow_pickle=False) as f: h=f['CTH_raw'];clp=f['CLP_raw'];cot=f['COT_raw']
    valid=np.isfinite(h)&np.isfinite(cot)&np.isin(clp,[1,2])&(h>0)&(h<=17.5)
    if not valid.any():
        records.append(dict(time_bjt=frame['time_bjt'],status='no_valid_cloud_pixels'))
        continue
    rr,cc=np.nonzero(valid);la=lat[valid];lo=lon[valid];height=h[valid]*1000.
    dt=datetime.strptime(frame['time_bjt'],'%Y%m%d%H%M')-timedelta(hours=8)
    zen,az,day=solar.compute_solar_angles(la,lo,dt)
    assert (day>0).all() and (zen<80).all()
    sun=ray.sun_ecef_from_local_angles(la,lo,90.-zen,az)
    top_nominal=ray.nominal_to_cloud_top(la,lo,height)
    top_corrected=ray.geodetic_to_ecef(la,lo,height)
    for scenario,top in [('nominal_geolocation',top_nominal),('already_corrected_geolocation',top_corrected)]:
        shadow=ray.cast_shadow(top,sun)
        slat,slon=ray.ecef_to_geodetic(shadow)
        assert np.isfinite(slat).all() and np.isfinite(slon).all()
        for st,(y,x) in centers.items():
            ex,ny=ray.local_xy(slat,slon,float(lat[y,x]),float(lon[y,x]));i=int(np.argmin(np.hypot(ex,ny)))
            records.append(dict(time_bjt=frame['time_bjt'],status='geometric_nearest_cloud_pixel',station=st,scenario=scenario,
                candidate_count=int(len(rr)),source_row=int(rr[i]),source_col=int(cc[i]),row_offset=int(rr[i]-y),col_offset=int(cc[i]-x),
                source_inside_3x3=bool(abs(rr[i]-y)<=1 and abs(cc[i]-x)<=1),
                nearest_shadow_center_km=float(np.hypot(ex[i],ny[i])),source_CTH_km=float(h[rr[i],cc[i]]),
                source_COT=float(cot[rr[i],cc[i]]),station_COT=float(cot[y,x]) if np.isfinite(cot[y,x]) else None,
                station_CTH_km=float(h[y,x]) if np.isfinite(h[y,x]) else None))
summary={'state':'HISTORICAL_CLOUD_HEIGHT_SENSITIVITY_DIAGNOSTIC','records':records,
    'date':'2025-08-06','scope':'single train initiation, seven historical snapshots, three sites',
    'assumptions':['nominal scan timestamp approximated by filename','source QA absent','two alternative geolocation assumptions, neither source lineage certified',
        'nearest cloud shadow point only; no cloud footprint or radiative transfer','CLP 1/2 treated as cloud; clear sky excluded from nearest candidates'],
    'future_cth_used':False,'release_time_verified':False,'test_payloads_read':0,'GHI_scored':False,
    'conclusion_boundary':'descriptive geometry only, not proof of impact or forecast skill'}
(P/'ACTUAL_SHADOW_SCENARIOS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (P/'ACTUAL_SHADOW_SCENARIOS.csv').open('w',newline='',encoding='utf-8') as f:
    keys=sorted(set().union(*(r.keys() for r in records)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
print('SCENARIOS_COMPLETE',len(records),'records')
