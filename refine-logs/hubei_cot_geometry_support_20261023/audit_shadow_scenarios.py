"""Independently replay saved 42 selected point geometries and payload values."""
import hashlib,importlib.util,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent;R=P.parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ray=module('rayaudit',R/'experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/project_cth_shadows.py')
solar=module('solaraudit',R/'refine-logs/hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/solar_geometry_noaa.py')
with np.load(R/'refine-logs/hubei_history_cot_confirmation_20261022/grid_static256.npz') as g:lat=g['lat'].astype(float);lon=g['lon'].astype(float)
centers={'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}
raw=(P/'ACTUAL_SHADOW_SCENARIOS.json').read_bytes();records=json.loads(raw)['records'];errors=[];backerrors=[];n=0
sat=np.array([ray.SAT_RADIUS_M*np.cos(np.deg2rad(105.)),ray.SAT_RADIUS_M*np.sin(np.deg2rad(105.)),0.])
for rec in records:
    if rec['status']!='geometric_nearest_cloud_pixel':continue
    i,j=rec['source_row'],rec['source_col'];sy,sx=centers[rec['station']]
    with np.load(P/'height_extract_attempt1'/(rec['time_bjt']+'.npz')) as z:
        h=float(z['CTH_raw'][i,j]);c=float(z['COT_raw'][i,j]);assert z['CLP_raw'][i,j] in [1,2]
        assert h==rec['source_CTH_km'] and c==rec['source_COT'] and float(z['COT_raw'][sy,sx])==rec['station_COT']
    assert (i-sy,j-sx)==(rec['row_offset'],rec['col_offset'])
    la=np.array(lat[i,j]);lo=np.array(lon[i,j]);height=np.array(h*1000.)
    utc=datetime.strptime(rec['time_bjt'],'%Y%m%d%H%M')-timedelta(hours=8)
    zen,az,_=solar.compute_solar_angles(la,lo,utc);sun=ray.sun_ecef_from_local_angles(la,lo,90-zen,az)
    if rec['scenario']=='nominal_geolocation':
        top=ray.nominal_to_cloud_top(la,lo,height)
        back=ray.ellipsoid_intersection(top,top-sat,0.)
        backerrors.append(float(np.linalg.norm(back-ray.geodetic_to_ecef(la,lo,0.))))
    else:
        assert rec['scenario']=='already_corrected_geolocation'
        top=ray.geodetic_to_ecef(la,lo,height)
    projected=ray.cast_shadow(top,sun);pla,plo=ray.ecef_to_geodetic(projected)
    dx,dy=ray.local_xy(pla,plo,float(lat[sy,sx]),float(lon[sy,sx]))
    errors.append(abs(float(np.hypot(dx,dy))-rec['nearest_shadow_center_km']));n+=1
assert n==42 and max(errors)<1e-8 and max(backerrors)<0.1
out={'state':'SELECTED_POINTS_REPLAY_PASS','selected_points':n,'nominal_satellite_backprojection_count':len(backerrors),
     'max_distance_difference_km':max(errors),'max_satellite_backprojection_error_m':max(backerrors),
     'source_json_sha256':hashlib.sha256(raw).hexdigest(),
     'limits':'Validates selected point calculations, not nearest-point search, QA, source lineage, or GHI benefit.'}
(P/'SHADOW_POINT_REPLAY.json').write_text(json.dumps(out,indent=2))
print(out)
